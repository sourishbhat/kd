from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Camera only
    #
    # namespace='camera' -> publishes /camera/image_raw and /camera/camera_info,
    # which match the default topics in whycode.launch.
    # camera_info_url + camera_name make usb_cam load YOUR committed calibration,
    # so /camera/camera_info carries the real intrinsic matrix (not zeros).
    return LaunchDescription([
        Node(
            package='usb_cam',
            name='usb_cam',
            namespace='camera',
            executable='usb_cam_node_exe',
            output='screen',
            parameters=[{
                'video_device': '/dev/video4',
                'image_width': 1920,
                'image_height': 1080,
                'pixel_format': 'mjpeg2rgb',
                'io_method': 'mmap',
                'framerate': 30.0,
                'camera_frame_id': 'usb_cam',
                'av_device_format': 'YUV422P'  # must match camera_name in the yaml
            }]
        ),
    ])
