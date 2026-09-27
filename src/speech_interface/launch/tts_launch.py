from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch file for TTS node
    """
    return LaunchDescription([
        Node(
            package='speech_interface',
            executable='tts_node',
            name='tts_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                # Add parameters here if needed in future
            ]
        )
    ])