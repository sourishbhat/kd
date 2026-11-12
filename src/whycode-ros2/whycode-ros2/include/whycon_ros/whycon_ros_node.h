#ifndef WHYCODEROS2_CWHYCONROSNODE_H
#define WHYCODEROS2_CWHYCONROSNODE_H

#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include <whycode_interfaces/srv/select_marker.hpp>
#include <whycode_interfaces/srv/set_calib_method.hpp>
#include <whycode_interfaces/srv/set_calib_path.hpp>
#include <whycode_interfaces/srv/set_coords.hpp>
#include <whycode_interfaces/srv/set_drawing.hpp>
#include <whycode_interfaces/srv/get_gui_settings.hpp>

#include <whycode_interfaces/msg/marker_array.hpp>
#include <whycode_interfaces/msg/marker.hpp>

#include "whycon/whycon.h"
#include "whycon_ros/shm_image.hpp"


namespace whycode_ros2
{

class CWhyconROSNode : public rclcpp::Node
{

    public:
        void getGuiSettingsCallback(const std::shared_ptr<whycode_interfaces::srv::GetGuiSettings::Request> req,
                                          std::shared_ptr<whycode_interfaces::srv::GetGuiSettings::Response> res);

        void setDrawingCallback(const std::shared_ptr<whycode_interfaces::srv::SetDrawing::Request> req,
                                      std::shared_ptr<whycode_interfaces::srv::SetDrawing::Response> res);

        void setCoordsCallback(const std::shared_ptr<whycode_interfaces::srv::SetCoords::Request> req,
                                     std::shared_ptr<whycode_interfaces::srv::SetCoords::Response> res);

        void setCalibMethodCallback(const std::shared_ptr<whycode_interfaces::srv::SetCalibMethod::Request> req,
                                          std::shared_ptr<whycode_interfaces::srv::SetCalibMethod::Response> res);

        void setCalibPathCallback(const std::shared_ptr<whycode_interfaces::srv::SetCalibPath::Request> req,
                                        std::shared_ptr<whycode_interfaces::srv::SetCalibPath::Response> res);

        void selectMarkerCallback(const std::shared_ptr<whycode_interfaces::srv::SelectMarker::Request> req,
                                        std::shared_ptr<whycode_interfaces::srv::SelectMarker::Response> res);

        void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

        void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);

        // Polls the shared-memory frame ring. Runs as a wall timer on the
        // executor thread -- deliberately NOT on a thread of its own: every
        // other path that touches whycon_ (cameraInfoCallback and the six
        // reconfiguration services) already runs there, so staying on it keeps
        // the detector single-threaded and needs no locking at all. A poll
        // that finds nothing new is one atomic load, so polling several times
        // per frame costs nothing worth measuring.
        void shmPollCallback();

        // The half of imageCallback that is transport-independent: runs the
        // detector over whatever is currently in image_ and publishes the
        // markers (and the annotated image, if use_gui_). Both the DDS
        // callback and the shm poll funnel into this.
        void processCurrentImage(const std_msgs::msg::Header &header);

        CWhyconROSNode();

        ~CWhyconROSNode();

    private:

        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr      cam_info_sub_;
        rclcpp::Publisher<whycode_interfaces::msg::MarkerArray>::SharedPtr markers_pub_;

        image_transport::Subscriber img_sub_;
        image_transport::Publisher  img_pub_;

        // Shared-memory frame source. When img_shm_name is set, the ring
        // REPLACES the image_transport subscription above (img_sub_ is never
        // created) -- see the rationale in shm_image.hpp. The annotated
        // ~/processed_image output stays on DDS either way: it feeds
        // image_view, which is a human display and not in any control path.
        shm_image::Reader shm_reader_;
        rclcpp::TimerBase::SharedPtr shm_timer_;
        std::string shm_frame_id_;
        bool use_shm_ = false;
        bool shm_logged_attach_ = false;
        
        rclcpp::Service<whycode_interfaces::srv::GetGuiSettings>::SharedPtr gui_settings_srv_;
        rclcpp::Service<whycode_interfaces::srv::SetDrawing>::SharedPtr     drawing_srv_;
        rclcpp::Service<whycode_interfaces::srv::SetCoords>::SharedPtr      coord_system_srv_;
        rclcpp::Service<whycode_interfaces::srv::SetCalibMethod>::SharedPtr calib_method_srv_;
        rclcpp::Service<whycode_interfaces::srv::SetCalibPath>::SharedPtr   calib_path_srv_;
        rclcpp::Service<whycode_interfaces::srv::SelectMarker>::SharedPtr   select_marker_srv_;



        bool draw_coords_;
        bool use_gui_;          // generate images for graphic interface?
        whycon::CWhycon whycon_;        // WhyCon instance
        whycon::CRawImage *image_;      // image wrapper for WhyCon
        double circle_diameter_;  // marker diameter [m]

        std::vector<whycon::SMarker> whycon_detections_;    // array of detected markers
        
        std::vector<float> intrinsic_mat_;        // intrinsic matrix from camera_info topic
        std::vector<float> distortion_coeffs_;    // distortion parameters from camera_info topic

        bool identify_;
        int num_markers_;
        int min_size_;
};

}  // namespace whycode_ros2


#endif  // WHYCODEROS2_CWHYCONROSNODE_H
