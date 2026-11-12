#include "whycon_ros/whycon_ros_node.h"

#include <functional>
#include <memory>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

using std::placeholders::_1;
using std::placeholders::_2;

namespace whycode_ros2
{

void CWhyconROSNode::getGuiSettingsCallback(const std::shared_ptr<whycode_interfaces::srv::GetGuiSettings::Request> req,
                                                  std::shared_ptr<whycode_interfaces::srv::GetGuiSettings::Response> res)
{
    RCLCPP_INFO(this->get_logger(), "getGuiSettingsCallback");
    res->draw_coords = whycon_.getDrawCoords();
    res->draw_segments = whycon_.getDrawSegments();
    res->coords = whycon_.getCoordinates();
}

void CWhyconROSNode::setDrawingCallback(const std::shared_ptr<whycode_interfaces::srv::SetDrawing::Request> req,
                                              std::shared_ptr<whycode_interfaces::srv::SetDrawing::Response> res)
{
    RCLCPP_INFO(this->get_logger(), "setDrawingCallback coords %d segs %d", req->draw_coords, req->draw_segments);
    whycon_.setDrawing(req->draw_coords, req->draw_segments);
    res->success = true;
}

void CWhyconROSNode::setCoordsCallback(const std::shared_ptr<whycode_interfaces::srv::SetCoords::Request> req,
                                             std::shared_ptr<whycode_interfaces::srv::SetCoords::Response> res)
{
    RCLCPP_INFO(this->get_logger(), "setCoordsCallback %d", req->coords);
    try
    {
        whycon_.setCoordinates(static_cast<whycon::ETransformType>(req->coords));
        res->success = true;
    }
    catch(const std::exception& e)
    {
        res->success = false;
        res->msg = e.what();
    }
}

void CWhyconROSNode::setCalibMethodCallback(const std::shared_ptr<whycode_interfaces::srv::SetCalibMethod::Request> req,
                                                  std::shared_ptr<whycode_interfaces::srv::SetCalibMethod::Response> res)
{
    RCLCPP_INFO(this->get_logger(), "setCalibMethodCallback %d", req->method);
    try
    {
        if(req->method == 0)
        {
            whycon_.autocalibration();
            res->success = true;
        }
        else if(req->method == 1)
        {
            whycon_.manualcalibration();
            res->success = true;
        }
        else
        {
            res->success = false;
            res->msg = "ERROR in setting calibration method : unkown method '" + std::to_string(req->method) + "'";
        }
    }
    catch(const std::exception& e)
    {
        res->success = false;
        res->msg = e.what();
    }
}

void CWhyconROSNode::setCalibPathCallback(const std::shared_ptr<whycode_interfaces::srv::SetCalibPath::Request> req,
                                                std::shared_ptr<whycode_interfaces::srv::SetCalibPath::Response> res)
{
    RCLCPP_INFO(this->get_logger(), "setCalibPathCallback action %s path %s", req->action.c_str(), req->path.c_str());
    try
    {
        if(req->action == "load")
        {
            whycon_.loadCalibration(req->path);
            res->success = true;
        }
        else if(req->action == "save")
        {
            whycon_.saveCalibration(req->path);
            res->success = true;
        }
        else
        {
            res->success = false;
            res->msg = "ERROR in setting calibration path : unkown action '" + req->action + "'";
        }
    }
    catch(const std::exception& e)
    {
        res->success = false;
        res->msg = e.what();
    }
}

void CWhyconROSNode::selectMarkerCallback(const std::shared_ptr<whycode_interfaces::srv::SelectMarker::Request> req,
                                                std::shared_ptr<whycode_interfaces::srv::SelectMarker::Response> res)
{
    RCLCPP_INFO(this->get_logger(), "selectMarkerCallback x %f y %f", req->point.x, req->point.y);
    whycon_.selectMarker(req->point.x, req->point.y);
}

/* void CWhyconROSNode::reconfigureCallback(whycon::whyconConfig& config, uint32_t level)
{
    ROS_INFO("[Reconfigure Request]\n"
        "identify %s circle_diameter %lf num_markers %d\n"
        "min_size %d field_length %lf field_width %lf\n"
        "initial_circularity_tolerance %lf final_circularity_tolerance %lf\n"
        "area_ratio_tolerance %lf\n"
        "center_distance_tolerance_ratio %lf center_distance_tolerance_abs %lf\n",
        (config.identify) ? "True" : "False",
        config.circle_diameter, config.num_markers, config.min_size, config.field_length,
        config.field_width, config.initial_circularity_tolerance,
        config.final_circularity_tolerance, config.area_ratio_tolerance,
        config.center_distance_tolerance_ratio, config.center_distance_tolerance_abs
        );

    whycon_.updateConfiguration(
        config.identify, config.circle_diameter, config.num_markers, config.min_size,
        config.field_length, config.field_width, config.initial_circularity_tolerance,
        config.final_circularity_tolerance, config.area_ratio_tolerance,
        config.center_distance_tolerance_ratio, config.center_distance_tolerance_abs
        );

    identify_ = config.identify;
} */

void CWhyconROSNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    if(msg->k[0] == 0)
    {
        RCLCPP_ERROR_ONCE(this->get_logger(), "ERROR: Camera is not calibrated!");
        return;
    }

    if(!std::equal(intrinsic_mat_.begin(), intrinsic_mat_.end(), msg->k.begin()))
        intrinsic_mat_.assign(msg->k.begin(), msg->k.end());

    if(!std::equal(distortion_coeffs_.begin(), distortion_coeffs_.end(), msg->d.begin()))
        distortion_coeffs_.assign(msg->d.begin(), msg->d.end());

    whycon_.updateCameraInfo(intrinsic_mat_, distortion_coeffs_);
}

void CWhyconROSNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
    image_->updateImage((unsigned char*)&msg->data[0], msg->width, msg->height, msg->step / msg->width);
    processCurrentImage(msg->header);
}

// Reads the newest frame out of the shared-memory ring, if there is one that
// is newer than the last one processed, and runs the detector over it.
//
// Note there is no catch-up loop here: read_latest hands back only the MOST
// RECENT completed frame and silently skips anything the producer wrote while
// the detector was busy. That is the correct behaviour for vision in a control
// loop -- processing a backlog would mean feeding the controller positions the
// drone held several frames ago, which is worse than feeding it nothing.
void CWhyconROSNode::shmPollCallback()
{
    if(!shm_reader_.try_attach())
    {
        return;  // producer not up yet -- keep polling, this is not an error
    }

    if(!shm_logged_attach_)
    {
        shm_logged_attach_ = true;
        RCLCPP_INFO(this->get_logger(), "Attached to frame ring: %dx%d, %d channels.",
                    shm_reader_.width(), shm_reader_.height(), shm_reader_.channels());
    }

    // Size image_ to the ring's geometry ONCE, so that from here on the frame
    // can be copied straight out of shared memory into the detector's own
    // buffer -- no staging vector, no second copy. CRawImage::updateImage is
    // only used for its (re)allocation side effect; the zeros it copies are
    // overwritten by the very next read_latest.
    const int w = shm_reader_.width();
    const int h = shm_reader_.height();
    const int c = shm_reader_.channels();
    if(image_->width_ != w || image_->height_ != h || image_->bpp_ != c)
    {
        std::vector<unsigned char> blank(static_cast<std::size_t>(w) * h * c, 0);
        image_->updateImage(blank.data(), w, h, c);
    }

    double sim_time = 0.0;
    if(!shm_reader_.read_latest(image_->data_, static_cast<std::size_t>(image_->size_), &sim_time))
    {
        return;  // nothing new since last poll -- the common case
    }

    // The producer stamps each slot with the sim time of the PIXELS, which its
    // pipelined readback makes one render period older than the sim state at
    // the moment of the write. Carry it through unchanged: re-stamping with
    // now() here would throw away that correction and hand the controller a
    // phase error.
    std_msgs::msg::Header header;
    header.stamp.sec = static_cast<int32_t>(sim_time);
    header.stamp.nanosec = static_cast<uint32_t>((sim_time - header.stamp.sec) * 1e9);
    header.frame_id = shm_frame_id_;
    processCurrentImage(header);
}

void CWhyconROSNode::processCurrentImage(const std_msgs::msg::Header &header)
{
    whycon_.processImage(image_, whycon_detections_);

    whycode_interfaces::msg::MarkerArray marker_array;
    marker_array.header.stamp = header.stamp;
    marker_array.header.frame_id = header.frame_id;

    for(const whycon::SMarker &detection : whycon_detections_)
    {
        whycode_interfaces::msg::Marker marker;

        marker.id = detection.seg.ID;
        marker.size = detection.seg.size;
        marker.u = detection.seg.x;
        marker.v = detection.seg.y;
        marker.angle = detection.obj.angle;

        marker.position.position.x = detection.obj.x;
        marker.position.position.y = detection.obj.y;
        marker.position.position.z = detection.obj.z;
        marker.position.orientation.x = detection.obj.qx;
        marker.position.orientation.y = detection.obj.qy;
        marker.position.orientation.z = detection.obj.qz;
        marker.position.orientation.w = detection.obj.qw;
        marker.rotation.x = detection.obj.roll;
        marker.rotation.y = detection.obj.pitch;
        marker.rotation.z = detection.obj.yaw;
        marker_array.markers.push_back(marker);
    }

    if(marker_array.markers.size() > 0)
    {
        markers_pub_->publish(marker_array);
    }

    if(use_gui_)
    {
        // Built from image_ rather than from an input message, so this works
        // identically whichever transport the frame arrived on. image_->data_
        // is what the detector just drew its overlays into.
        const size_t step = static_cast<size_t>(image_->width_) * image_->bpp_;
        sensor_msgs::msg::Image out_msg;
        out_msg.header = header;
        out_msg.height = image_->height_;
        out_msg.width = image_->width_;
        out_msg.encoding = image_->bpp_ == 1 ? "mono8" : "rgb8";
        out_msg.step = static_cast<uint32_t>(step);
        out_msg.data.resize(step * image_->height_);
        std::memcpy((void*)&out_msg.data[0], image_->data_, step * image_->height_);
        img_pub_.publish(out_msg);
    }

    whycon_detections_.clear();
}

CWhyconROSNode::CWhyconROSNode() :
    intrinsic_mat_(9),
    distortion_coeffs_(5),
    Node("whycon")
{
    int id_bits;
    int id_samples;
    int hamming_dist;
    std::string calib_path;
    int coords_method;
    std::string img_base_topic;
    std::string img_transport;
    std::string info_topic;

    this->declare_parameter("use_gui", true);
    this->declare_parameter("circle_diameter", 0.122);
    this->declare_parameter("id_bits", 6);
    this->declare_parameter("id_samples", 360);
    this->declare_parameter("hamming_dist", 1);
    this->declare_parameter("num_markers", 10);
    this->declare_parameter("calib_file", std::string(""));
    this->declare_parameter("coords_method", 0);
    this->declare_parameter("min_size", 20);
    this->declare_parameter("img_base_topic", std::string(""));
    this->declare_parameter("img_transport", std::string(""));
    this->declare_parameter("info_topic", std::string(""));

    // Shared-memory frame source. Empty (the default) keeps the original
    // image_transport subscription, so an existing launch file that does not
    // set this behaves exactly as before. Setting it to the producer's ring
    // name switches transports entirely -- see shm_image.hpp for why.
    this->declare_parameter("img_shm_name", std::string(""));
    // Poll rate for that ring. Well above the frame rate on purpose: the ring
    // has no way to wake us, so this sets the worst-case added latency
    // (1/poll_hz). At 200Hz that is 5ms against a ~30Hz frame period, and a
    // poll that finds nothing is a single atomic load.
    this->declare_parameter("img_shm_poll_hz", 200.0);
    // The ring carries pixels and a timestamp but no frame_id, so the one to
    // stamp markers with is named here. Must match what the producer would
    // have put on /image_raw or downstream TF lookups break.
    this->declare_parameter("img_shm_frame_id", std::string("camera_optical"));

    use_gui_ = this->get_parameter("use_gui").as_bool();
    circle_diameter_ = this->get_parameter("circle_diameter").as_double();
    id_bits = this->get_parameter("id_bits").as_int();
    id_samples = this->get_parameter("id_samples").as_int();
    hamming_dist = this->get_parameter("hamming_dist").as_int();
    num_markers_ = this->get_parameter("num_markers").as_int();
    calib_path = this->get_parameter("calib_file").as_string();
    coords_method = this->get_parameter("coords_method").as_int();
    min_size_ = this->get_parameter("min_size").as_int();
    img_base_topic = this->get_parameter("img_base_topic").as_string();
    img_transport = this->get_parameter("img_transport").as_string();
    info_topic = this->get_parameter("info_topic").as_string();

    const std::string img_shm_name = this->get_parameter("img_shm_name").as_string();
    double img_shm_poll_hz = this->get_parameter("img_shm_poll_hz").as_double();
    shm_frame_id_ = this->get_parameter("img_shm_frame_id").as_string();
    use_shm_ = !img_shm_name.empty();
    if(img_shm_poll_hz <= 0.0)
    {
        RCLCPP_WARN(this->get_logger(), "img_shm_poll_hz must be > 0; using 200.");
        img_shm_poll_hz = 200.0;
    }

    
    int default_width = 640;
    int default_height = 480;
    image_ = new whycon::CRawImage(default_width, default_height, 3);
    whycon_.init(circle_diameter_, use_gui_, id_bits, id_samples, hamming_dist, num_markers_, default_width, default_height);
    


    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(info_topic, 1, std::bind(&CWhyconROSNode::cameraInfoCallback, this, _1));
    markers_pub_ = this->create_publisher<whycode_interfaces::msg::MarkerArray>("~/markers", 1);

    if(use_shm_)
    {
        // The ring replaces the subscription outright. Creating both would
        // mean the detector ran twice per frame on two copies of the same
        // pixels, which is exactly the cost this path exists to remove.
        shm_reader_.configure(img_shm_name);
        shm_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / img_shm_poll_hz),
            std::bind(&CWhyconROSNode::shmPollCallback, this));
        RCLCPP_INFO(this->get_logger(),
                    "Frames from shared-memory ring '%s' (polled at %.0f Hz); "
                    "not subscribing to '%s'.",
                    img_shm_name.c_str(), img_shm_poll_hz, img_base_topic.c_str());
    }
    else
    {
        // "image_transport" parameter can chagne the transport during startup. default is "raw" (see image_transport::TransportHints)
        img_sub_ = image_transport::create_subscription(this, img_base_topic, std::bind(&CWhyconROSNode::imageCallback, this, _1), img_transport);
    }
    img_pub_ = image_transport::create_publisher(this, "~/processed_image");


    
    gui_settings_srv_ = this->create_service<whycode_interfaces::srv::GetGuiSettings>("~/get_gui_settings", std::bind(&CWhyconROSNode::getGuiSettingsCallback, this, _1, _2));
    drawing_srv_ = this->create_service<whycode_interfaces::srv::SetDrawing>("~/set_drawing", std::bind(&CWhyconROSNode::setDrawingCallback, this, _1, _2));
    coord_system_srv_ = this->create_service<whycode_interfaces::srv::SetCoords>("~/set_coords", std::bind(&CWhyconROSNode::setCoordsCallback, this, _1, _2));
    calib_method_srv_ = this->create_service<whycode_interfaces::srv::SetCalibMethod>("~/set_calib_method", std::bind(&CWhyconROSNode::setCalibMethodCallback, this, _1, _2));
    calib_path_srv_ = this->create_service<whycode_interfaces::srv::SetCalibPath>("~/set_calib_path", std::bind(&CWhyconROSNode::setCalibPathCallback, this, _1, _2));
    select_marker_srv_ = this->create_service<whycode_interfaces::srv::SelectMarker>("~/select_marker", std::bind(&CWhyconROSNode::selectMarkerCallback, this, _1, _2));



    identify_ = true;
    double field_length = 1.0;
    double field_width = 1.0;
    double initial_circularity_tolerance = 100.0;
    double final_circularity_tolerance = 2.0;
    double area_ratio_tolerance = 40.0;
    double center_distance_tolerance_ratio = 10.0;
    double center_distance_tolerance_abs = 5.0;
    whycon_.updateConfiguration(
        identify_, circle_diameter_, num_markers_, min_size_,
        field_length, field_width, initial_circularity_tolerance,
        final_circularity_tolerance, area_ratio_tolerance,
        center_distance_tolerance_ratio, center_distance_tolerance_abs
        );

    if(calib_path.size() > 0)
    {
        try
        {
            whycon_.loadCalibration(calib_path);
            whycon_.setCoordinates(static_cast<whycon::ETransformType>(coords_method));
        }
        catch(const std::exception& e)
        {
            RCLCPP_WARN(this->get_logger(), "Calibration file '%s' could not be loaded. Using camera centric coordinates.", calib_path.c_str());
        }
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Calibration file path empty. Using camera centric coordinates.");
    }
}

CWhyconROSNode::~CWhyconROSNode()
{
    delete image_;
}

}  // namespace whycode_ros2

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<whycode_ros2::CWhyconROSNode>());
    rclcpp::shutdown();

    return 0;
}
