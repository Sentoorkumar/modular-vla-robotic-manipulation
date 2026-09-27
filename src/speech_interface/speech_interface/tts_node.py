#!/usr/bin/env python3

"""
ROS2 Piper TTS Node
Offline neural text-to-speech for robotic interaction
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import subprocess
import threading
import os
import tempfile


class TTSNode(Node):

    def __init__(self):
        super().__init__('tts_node')

        # Path to Piper model
        self.model_path = os.path.expanduser(
            '~/.local/share/piper-voices/en_US-lessac-medium.onnx'
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Piper model not found: {self.model_path}")

        # Serialize speech: overlapping /feedback/text messages must not run
        # Piper + aplay concurrently, so each speak() call holds this lock.
        self.lock = threading.Lock()

        self.subscription = self.create_subscription(
            String,
            '/feedback/text',
            self.callback,
            10
        )

        self.get_logger().info("Piper TTS node ready.")

    def callback(self, msg: String):
        text = msg.data.strip()
        if text:
            self.speak(text)

    def speak(self, text: str):
        with self.lock:
            try:
                # Create temp WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name

                # Generate speech
                process = subprocess.Popen(
                    [
                        "piper",
                        "--model", self.model_path,
                        "--output_file", wav_path
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True
                )

                process.communicate(input=text)

                if process.returncode != 0:
                    self.get_logger().error("Piper generation failed.")
                    return

                # Play audio
                subprocess.run(
                    ["aplay", "-q", wav_path],
                    check=True
                )

                os.remove(wav_path)

            except Exception as e:
                self.get_logger().error(f"TTS error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
