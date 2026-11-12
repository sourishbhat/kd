from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # --- Camera: publishes /camera/image_raw + calibrated /camera/camera_info ---
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
                'av_device_format': 'YUV422P',
                'camera_name': 'narrow_stereo',
                'camera_info_url': 'file://$(env HOME)/.ros/camera_info/camera_info.yaml',
            }]
        ),

        # --- Detector: draws the overlay onto /whycode_node/processed_image ---
        Node(
            package='whycode',
            executable='whycode_node',
            name='whycode_node',
            output='screen',
            parameters=[{
                'img_base_topic': '/camera/image_raw',
                'info_topic': '/camera/camera_info',
                'img_transport': 'raw',
                'circle_diameter': 0.05,   # Marker's outer black-ring diameter (m)
                'id_bits': 3,
                'id_samples': 720,
                'hamming_dist': 1,
                'num_markers': 1,         # max markers to detect simultaneously
                'use_gui': True,           # MUST be true, or processed_image is not published
                'min_size': 20,
                'calib_file': '',          # empty = camera-relative pose
                'coords_method': 0,
            }]
        ),

        # --- Viewer: auto-opens a window showing the overlay (like the reference repo) ---
        Node(
            package='image_view',
            executable='image_view',
            name='image_view',
            namespace='whycode_display',
            output='screen',
            remappings=[('image', '/whycode_node/processed_image')],
        ),
    ])
