#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import time
import threading
import sys
import os

# Only import termios on Unix-like systems
try:
    import termios
    def clear_stdin():
        """Flush leftover input e.g., extra ENTER presses."""
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
except ImportError:
    def clear_stdin():
        pass

class STTNode(Node):
    def __init__(self, model_size="small", record_seconds=7):
        super().__init__("stt_node")

        # Parameters
        self.declare_parameter('model_size', model_size)
        self.declare_parameter('record_seconds', record_seconds)

        self.sample_rate = 16000 # Whisper requires 16 kHz input do not change
         # 7 s recording window — empirically chosen to fit a full command
        self.record_seconds = self.get_parameter('record_seconds').value
        self.wav_path = "/tmp/command.wav"
        self.is_recording = threading.Lock()

        # Publisher
        self.publisher_ = self.create_publisher(String, "/voice_cmd", 10)

        # Load model
        model_name = self.get_parameter('model_size').value
        self.get_logger().info(f"Loading Whisper model ({model_name})...")
        self.model = whisper.load_model(model_name)
        self.get_logger().info("Whisper model loaded.")

    def record_audio(self):
        """Record audio from the default microphone."""
        self.get_logger().info(f"Recording {self.record_seconds}s — speak now...")

        # Stop any ongoing playback/recording
        sd.stop()
        sd.wait()

        try:
            audio = sd.rec(
                int(self.record_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16"
            )
            sd.wait()

            # Save to file
            write(self.wav_path, self.sample_rate, audio)
            self.get_logger().info(f"Audio saved to {self.wav_path}")
            return True

        except Exception as e:
            self.get_logger().error(f"Recording failed: {e}")
            return False

    def transcribe_audio(self):
        """Transcribe the recorded audio file using Whisper."""
        if not os.path.exists(self.wav_path):
            self.get_logger().error("Audio file not found!")
            return "", 0.0

        self.get_logger().info("Starting transcription...")
        start_time = time.perf_counter()

        try:
            result = self.model.transcribe(
                self.wav_path,
                language="en",
                fp16=False  # CPU compatibility
            )
            end_time = time.perf_counter()
            elapsed = end_time - start_time

            text = result.get("text", "").strip()
            self.get_logger().info(f"Transcription: '{text}'")
            self.get_logger().info(f"Transcription time: {elapsed:.2f} seconds")

            return text, elapsed

        except Exception as e:
            self.get_logger().error(f"Transcription failed: {e}")
            return "", 0.0

    def publish_text(self, text):
        """Publish transcribed text to ROS topic."""
        if not text:
            self.get_logger().warn("Empty transcription — nothing published.")
            return False

        msg = String()
        msg.data = text
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published to /voice_cmd: '{text}'")
        return True

def main(args=None):
    rclpy.init(args=args)
    node = STTNode(model_size="small", record_seconds=7)

    cooldown_time = 1.0

    print("\n Voice Command Node Ready")
    print("=" * 50)

    try:
        while rclpy.ok():
            clear_stdin()
            input("\n▶ Press ENTER to record (Ctrl+C to quit)\n")

            # Prevent concurrent recordings
            if not node.is_recording.acquire(blocking=False):
                print(" Already recording... please wait.")
                continue

            try:
                # Record → Transcribe → Publish pipeline
                if node.record_audio():
                    text, elapsed = node.transcribe_audio()

                    if text:
                        node.publish_text(text)
                        print(f"\n Recognized: '{text}'")
                        print(f" Time: {elapsed:.2f} sec")
                    else:
                        print("No speech detected")

                # Cooldown before next recording
                time.sleep(cooldown_time)

            finally:
                node.is_recording.release()

    except KeyboardInterrupt:
        print("\n\n Shutting down STT node...")

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

