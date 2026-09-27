#!/usr/bin/env python3
"""
Perception Node — Object Detection with Disambiguation
======================================================
Runs YOLO on the camera image, projects each detection to a 3D table
position (PixelToWorldConverter), assigns unique per-class labels sorted
by distance from the robot base (nearest = label 1), checks reachability,
flags same-class ambiguity, and publishes the full DetectedObject[] in
VisionResult.

Subscribes:  /camera/image_raw  (sensor_msgs/Image from Isaac Sim)
Publishes:   /perception/result (robot_interfaces/VisionResult)
"""

import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from robot_interfaces.msg import VisionResult, DetectedObject
from ultralytics import YOLO
from collections import Counter

# Import our pixel-to-world converter (same package)
from perception_agent.pixel_to_world import PixelToWorldConverter


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # ----------------------------------------------------------
        # 1. Load camera config
        # ----------------------------------------------------------
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'config', 'camera_config.yaml'
        )
        # Also check common install locations
        if not os.path.exists(config_path):
            config_path = os.path.join(
                os.path.expanduser('~'),
                'llm_arm_ws', 'src', 'perception_agent',
                'config', 'camera_config.yaml'
            )
        if not os.path.exists(config_path):
            self.get_logger().error(
                f'camera_config.yaml not found! '
                f'Tried: {config_path}'
            )
            raise FileNotFoundError(f'Config not found: {config_path}')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.get_logger().info(
            f'Loaded config from: {config_path}'
        )

        # ----------------------------------------------------------
        # 2. Initialize pixel-to-world converter
        # ----------------------------------------------------------
        self.converter = PixelToWorldConverter(self.config)


        self.get_logger().info(
            f'PixelToWorldConverter ready '
            f'(fx={self.converter.fx:.1f}, fy={self.converter.fy:.1f})'
        )

        # ----------------------------------------------------------
        # Spatial rejection filter
        # ----------------------------------------------------------
        # Pickable objects are placed in the front working area only.
        # The region behind this Y line holds the bins, robot base, and
        # rear wall -> reject any detection there as a false positive.
        # -0.53 m was tuned empirically: it sits just in front of the bins/base
        # so real table objects pass but rear-scene false positives are cut.
        self.front_y_threshold = -0.53   # reject detections with Y < this

        rb = self.config['robot']['base_position']
        self.base_xy = (rb['x'], rb['y'])
        # 0.12 m: tuned so this circle around the base catches the robot's own
        # body/mount as a false positive without clipping nearby real objects.
        self.base_reject_radius = 0.12   # m, safety net for the arm body

        self.get_logger().info(
            f'Spatial filter: reject Y < {self.front_y_threshold}, '
            f'base radius {self.base_reject_radius}m'
        )

        # ----------------------------------------------------------
        # 3. Load YOLO model
        # ----------------------------------------------------------
        # NOTE: not portable — this path is hardcoded to ~/llm_arm_ws and has no
        # fallback (unlike camera_config.yaml above). Update it if the workspace
        # moves, or YOLO will fail to load the model.
        model_path = os.path.join(
            os.path.expanduser('~'),
            'llm_arm_ws', 'src', 'perception_agent',
            'models', 'best.pt'
        )
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.get_logger().info(
            f'Model loaded. Classes: {self.model.names}'
        )

        # Detection confidence threshold from config
        self.conf_threshold = self.config['detection']['confidence_threshold']

        # ----------------------------------------------------------
        # 4. CV Bridge
        # ----------------------------------------------------------
        self.bridge = CvBridge()

        # ----------------------------------------------------------
        # 5. ROS2 subscription and publisher
        # ----------------------------------------------------------
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.vision_pub = self.create_publisher(
            VisionResult,
            '/perception/result',
            10
        )

        self.frame_count = 0
        self.get_logger().info(
            'Perception node started — waiting for camera images.'
        )

    # ==============================================================
    # Main callback
    # ==============================================================

    def image_callback(self, msg):
        """Process incoming camera image from Isaac Sim."""
        self.frame_count += 1

        # Convert ROS Image -> OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        # Run YOLO
        results = self.model(
            cv_image, conf=self.conf_threshold, verbose=False
        )

        # Build list of raw detections
        raw_detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            raw_detections.append({
                'class_name': class_name,
                'confidence': confidence,
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
            })

        # Process detections: positions, labels, reachability
        detected_objects = self._process_detections(raw_detections)

        # Build and publish VisionResult
        vision_msg = self._build_vision_result(detected_objects, msg.header)
        self.vision_pub.publish(vision_msg)

        # Log summary
        if detected_objects:
            self.get_logger().info(
                f'Frame {self.frame_count}: '
                f'{vision_msg.scene_description} '
                f'(ambiguous={vision_msg.is_ambiguous})'
            )
        else:
            self.get_logger().info(
                f'Frame {self.frame_count}: No objects detected'
            )

    # ==============================================================
    # Processing pipeline
    # ==============================================================

    def _process_detections(self, raw_detections: list) -> list:
        """
        Takes raw YOLO detections and enriches them with:
          - 3D world position (from pixel-to-world math)
          - Pose (standing / lying)
          - Distance from robot
          - Reachability
          - Unique labels sorted by distance

        Returns a list of dicts, sorted by distance from robot base.
        """
        enriched = []

        for det in raw_detections:
            # 1. Pick grounding pixel based on pose
            pixel_u, pixel_v, pose = \
                PixelToWorldConverter.pick_grounding_pixel(
                    det['x1'], det['y1'], det['x2'], det['y2']
                )

            # 2. Convert pixel to world position
            world_pos = self.converter.pixel_to_world(pixel_u, pixel_v)
            if world_pos is None:
                # Ray didn't hit table (object in upper half of image,
                # or some edge case). Skip this detection.
                self.get_logger().warn(
                    f'Skipping {det["class_name"]}: pixel '
                    f'({pixel_u:.0f},{pixel_v:.0f}) has no '
                    f'table intersection'
                )
                continue

             # --- Spatial rejection filter ---
            if self._is_rejected_region(world_pos):
                self.get_logger().info(
                    f'FILTERED OUT {det["class_name"]} at '
                    f'({world_pos[0]:.3f},{world_pos[1]:.3f}) '
                    f'(robot base or bin region)'
                )
                continue

            self.get_logger().info(
                f'DEBUG: {det["class_name"]} bbox=({det["x1"]:.0f},{det["y1"]:.0f},'
                f'{det["x2"]:.0f},{det["y2"]:.0f}) '
                f'grounding_pixel=({pixel_u:.1f},{pixel_v:.1f}) '
                f'world=({world_pos[0]:.4f},{world_pos[1]:.4f},{world_pos[2]:.4f})'
            )

            # 3. Calculate distance and reachability
            distance = self.converter.distance_from_robot_base(world_pos)
            reachable = self.converter.is_reachable(world_pos)

            enriched.append({
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'x1': det['x1'], 'y1': det['y1'],
                'x2': det['x2'], 'y2': det['y2'],
                'pixel_u': pixel_u,
                'pixel_v': pixel_v,
                'world_pos': world_pos,
                'pose': pose,
                'distance': distance,
                'reachable': reachable,
            })


        # 4. Sort by distance from robot (nearest first)
        enriched.sort(key=lambda d: d['distance'])

        # 5. Assign unique labels per class
        #    e.g. "bottle 1" (nearest), "bottle 2" (next), ...
        class_counters = {}
        for det in enriched:
            cls = det['class_name']
            if cls not in class_counters:
                class_counters[cls] = 0
            class_counters[cls] += 1
            det['label'] = f"{cls} {class_counters[cls]}"

        return enriched

    def _is_rejected_region(self, world_pos) -> bool:
        """Reject false positives: detections behind the front working area
        (bins / base / rear wall), or coincident with the arm base itself.
        Reachability of valid objects is handled separately."""
        import math
        x, y = world_pos[0], world_pos[1]

        # Reject anything behind the front working area (bins / base / wall)
        if y < self.front_y_threshold:
            return True

        # Safety: tight radius around robot base catches the arm body
        dxb = x - self.base_xy[0]
        dyb = y - self.base_xy[1]
        if math.sqrt(dxb * dxb + dyb * dyb) <= self.base_reject_radius:
            return True

        return False

    # ==============================================================
    # Message building
    # ==============================================================

    def _build_vision_result(
        self, detected_objects: list, image_header
    ) -> VisionResult:
        """Build a complete VisionResult message."""

        msg = VisionResult()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'

        # --- Build DetectedObject array ---
        obj_msgs = []
        for det in detected_objects:
            obj = DetectedObject()
            obj.label = det['label']
            obj.object_class = det['class_name']
            obj.confidence = det['confidence']

            # 2D pixel info
            obj.pixel_u = det['pixel_u']
            obj.pixel_v = det['pixel_v']
            obj.bbox_x1 = det['x1']
            obj.bbox_y1 = det['y1']
            obj.bbox_x2 = det['x2']
            obj.bbox_y2 = det['y2']

            # 3D world info
            obj.world_x = float(det['world_pos'][0])
            obj.world_y = float(det['world_pos'][1])
            obj.world_z = float(det['world_pos'][2])

            # Pose
            obj.pose = det['pose']

            # Reachability
            obj.distance_from_robot = float(det['distance'])
            obj.reachable = det['reachable']
            obj.reach_status = 'reachable' if det['reachable'] \
                else 'too_far'


            obj_msgs.append(obj)

        msg.objects = obj_msgs
        msg.num_objects = len(obj_msgs)
        msg.num_reachable = sum(
            1 for o in obj_msgs if o.reachable
        )

        # --- Disambiguation info ---
        class_counts = Counter(
            det['class_name'] for det in detected_objects
        )
        ambiguous_classes = [
            cls for cls, count in class_counts.items() if count > 1
        ]
        msg.is_ambiguous = len(ambiguous_classes) > 0
        msg.ambiguous_classes = ambiguous_classes

        # --- Scene description ---
        msg.scene_description = self._generate_scene_description(
            detected_objects
        )

        # --- Legacy fields (backward compatibility) ---
        msg.detected_objects = [
            det['class_name'] for det in detected_objects
        ]
        msg.confidence_scores = [
            det['confidence'] for det in detected_objects
        ]
        msg.graspable = msg.num_reachable > 0
        if msg.graspable:
            msg.graspability_reason = (
                f'{msg.num_reachable} of {msg.num_objects} '
                f'objects within reach'
            )
        else:
            msg.graspability_reason = 'No objects within reach'
        msg.intent_type = ''
        msg.target_object = ''

        return msg

    # ==============================================================
    # Scene description generator
    # ==============================================================

    def _generate_scene_description(self, detected_objects: list) -> str:
        """
        Generate a natural-language scene description including:
        - Object counts and labels
        - Reachability info
        - Ambiguity note
        """
        if not detected_objects:
            return "No objects detected on the table."

        # Count per class
        class_counts = Counter(
            det['class_name'] for det in detected_objects
        )

        # Build count phrase: "2 bottles and 1 can"
        parts = []
        for cls, count in class_counts.items():
            plural = 's' if count > 1 else ''
            parts.append(f"{count} {cls}{plural}")
        if len(parts) == 1:
            count_phrase = parts[0]
        elif len(parts) == 2:
            count_phrase = f"{parts[0]} and {parts[1]}"
        else:
            count_phrase = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        # Build label list with reachability
        label_parts = []
        for det in detected_objects:
            reach = "reachable" if det['reachable'] else "too far"
            label_parts.append(
                f"{det['label']} ({reach}, "
                f"{det['distance']:.2f}m)"
            )

        desc = (
            f"I see {count_phrase}: "
            + ", ".join(label_parts) + "."
        )

        return desc


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()