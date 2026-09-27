#!/usr/bin/env python3
"""
Validation Node — Compare Predicted vs Ground Truth Positions
==============================================================
PURPOSE:
    Subscribes to:
        /perception/result    (predicted positions from perception node)
        /ground_truth/poses   (real positions from Isaac Sim)

    Compares them and outputs an error table showing accuracy.

MATCHING:
    Objects are matched by nearest-neighbor: for each predicted object,
    find the closest ground truth object of the same class.

THRESHOLD:
    Default: 20 mm (0.02 m) — objects within this error are "PASS".

USAGE:
    cd ~/llm_arm_ws
    source install/setup.bash
    python3 src/perception_agent/perception_agent/validation_node.py

OUTPUT:
    Prints a live error table to the terminal every 2 seconds.
    Also saves a CSV file for thesis use.

Author: Master's Thesis — LLM-Coordinated Pfand Sorting System
"""

import rclpy
from rclpy.node import Node
from robot_interfaces.msg import VisionResult
from std_msgs.msg import String
import json
import math
import time
import os
from datetime import datetime


class ValidationNode(Node):
    def __init__(self):
        super().__init__('validation_node')

        # ---- Config ----
        self.error_threshold_mm = 20.0  # PASS/FAIL threshold in mm
        self.print_interval = 2.0       # seconds between prints

        # ---- State ----
        self.latest_prediction = None
        self.latest_ground_truth = None
        self.last_print_time = 0.0

        # ---- Subscribers ----
        self.pred_sub = self.create_subscription(
            VisionResult,
            '/perception/result',
            self.prediction_callback,
            10
        )
        self.gt_sub = self.create_subscription(
            String,
            '/ground_truth/poses',
            self.ground_truth_callback,
            10
        )

        # ---- CSV output file ----
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.expanduser(
            f'~/llm_arm_ws/validation_results_{timestamp}.csv'
        )

        self.get_logger().info(
            f'Validation node started. '
            f'Threshold: {self.error_threshold_mm} mm'
        )
        self.get_logger().info(
            f'CSV output: {self.csv_path}'
        )
        self.get_logger().info(
            'Waiting for /perception/result and '
            '/ground_truth/poses...'
        )

    # ==============================================================
    # Callbacks
    # ==============================================================

    def prediction_callback(self, msg: VisionResult):
        """Store latest prediction from perception node."""
        pred_objects = []
        for obj in msg.objects:
            pred_objects.append({
                'label': obj.label,
                'class': obj.object_class,
                'world_x': obj.world_x,
                'world_y': obj.world_y,
            })
        self.latest_prediction = {
            'timestamp': time.time(),
            'objects': pred_objects,
        }
        self._try_compare()

    def ground_truth_callback(self, msg: String):
        """Store latest ground truth from Isaac Sim."""
        # NOTE: the ground-truth JSON must carry a top-level 'timestamp'
        # key; _try_compare() reads it for the freshness check.
        try:
            data = json.loads(msg.data)
            self.latest_ground_truth = data
        except json.JSONDecodeError as e:
            self.get_logger().error(
                f'Failed to parse ground truth JSON: {e}'
            )
        self._try_compare()

    # ==============================================================
    # Comparison logic
    # ==============================================================

    def _try_compare(self):
        """Compare if both data sources are fresh enough."""
        now = time.time()

        # Only print every N seconds to avoid spam
        if now - self.last_print_time < self.print_interval:
            return

        if self.latest_prediction is None:
            return
        if self.latest_ground_truth is None:
            return

        # Check data freshness (within 2 seconds)
        pred_age = now - self.latest_prediction['timestamp']
        gt_age = now - self.latest_ground_truth['timestamp']
        if pred_age > 2.0 or gt_age > 2.0:
            return

        self._compare_and_print()
        self.last_print_time = now

    def _compare_and_print(self):
        """Match predicted objects to ground truth and compute errors."""
        pred_objects = self.latest_prediction['objects']
        gt_data = self.latest_ground_truth

        # Filter ground truth: only visible (on-table) objects
        gt_objects = [
            o for o in gt_data['objects'] if o.get('visible', False)
        ]

        # Match each predicted object to nearest ground truth of same class
        matches = []
        used_gt_indices = set()

        for pred in pred_objects:
            best_match = None
            best_dist = float('inf')
            best_idx = -1

            for i, gt in enumerate(gt_objects):
                if i in used_gt_indices:
                    continue
                if gt['class'] != pred['class']:
                    continue

                # Euclidean distance in X-Y plane
                dx = pred['world_x'] - gt['world_x']
                dy = pred['world_y'] - gt['world_y']
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < best_dist:
                    best_dist = dist
                    best_match = gt
                    best_idx = i

            if best_match is not None:
                used_gt_indices.add(best_idx)
                error_mm = best_dist * 1000  # convert to mm
                passed = error_mm <= self.error_threshold_mm

                matches.append({
                    'pred_label': pred['label'],
                    'pred_class': pred['class'],
                    'pred_x': pred['world_x'],
                    'pred_y': pred['world_y'],
                    'gt_name': best_match['name'],
                    'gt_x': best_match['world_x'],
                    'gt_y': best_match['world_y'],
                    'error_mm': error_mm,
                    'passed': passed,
                })
            else:
                # No matching ground truth found
                matches.append({
                    'pred_label': pred['label'],
                    'pred_class': pred['class'],
                    'pred_x': pred['world_x'],
                    'pred_y': pred['world_y'],
                    'gt_name': 'NO MATCH',
                    'gt_x': 0.0,
                    'gt_y': 0.0,
                    'error_mm': -1,
                    'passed': False,
                })

        # Unmatched ground truth objects (missed detections)
        missed = []
        for i, gt in enumerate(gt_objects):
            if i not in used_gt_indices:
                missed.append(gt)

        # Print the comparison table
        self._print_table(matches, missed, pred_objects, gt_objects)

        # Save to CSV
        self._save_csv(matches)

    def _print_table(self, matches, missed, pred_objects, gt_objects):
        """Print a nice comparison table to terminal."""
        print()
        print("=" * 90)
        print(f"  VALIDATION REPORT  |  "
              f"Predicted: {len(pred_objects)}  |  "
              f"Ground Truth: {len(gt_objects)}  |  "
              f"Threshold: {self.error_threshold_mm} mm")
        print("=" * 90)
        print(f"  {'Predicted':15s} {'Pred (X,Y)':18s} "
              f"{'GT Name':22s} {'GT (X,Y)':18s} "
              f"{'Error':10s} {'Result':8s}")
        print("-" * 90)

        total_error = 0.0
        valid_count = 0

        for m in matches:
            if m['error_mm'] < 0:
                err_str = "N/A"
                res_str = "⚠️ MISS"
            else:
                err_str = f"{m['error_mm']:.1f} mm"
                res_str = "✅ PASS" if m['passed'] else "❌ FAIL"
                total_error += m['error_mm']
                valid_count += 1

            print(f"  {m['pred_label']:15s} "
                  f"({m['pred_x']:+.3f},{m['pred_y']:+.3f})   "
                  f"{m['gt_name']:22s} "
                  f"({m['gt_x']:+.3f},{m['gt_y']:+.3f})   "
                  f"{err_str:10s} {res_str}")

        for m in missed:
            print(f"  {'---':15s} {'---':18s} "
                  f"{m['name']:22s} "
                  f"({m['world_x']:+.3f},{m['world_y']:+.3f})   "
                  f"{'N/A':10s} ⚠️ UNDETECTED")

        print("-" * 90)
        if valid_count > 0:
            avg_err = total_error / valid_count
            max_err = max(
                m['error_mm'] for m in matches if m['error_mm'] >= 0
            )
            all_pass = all(
                m['passed'] for m in matches if m['error_mm'] >= 0
            )
            status = "✅ ALL PASS" if all_pass else "❌ SOME FAILED"
            print(f"  Avg error: {avg_err:.1f} mm  |  "
                  f"Max error: {max_err:.1f} mm  |  "
                  f"Status: {status}")
        else:
            print("  No valid matches to compare.")
        print("=" * 90)

    def _save_csv(self, matches):
        """Append comparison data to CSV file."""
        file_exists = os.path.exists(self.csv_path)

        with open(self.csv_path, 'a') as f:
            if not file_exists:
                f.write(
                    "timestamp,pred_label,pred_class,"
                    "pred_x,pred_y,"
                    "gt_name,gt_x,gt_y,"
                    "error_mm,passed\n"
                )
            for m in matches:
                f.write(
                    f"{time.time():.2f},"
                    f"{m['pred_label']},{m['pred_class']},"
                    f"{m['pred_x']:.5f},{m['pred_y']:.5f},"
                    f"{m['gt_name']},{m['gt_x']:.5f},"
                    f"{m['gt_y']:.5f},"
                    f"{m['error_mm']:.1f},"
                    f"{'PASS' if m['passed'] else 'FAIL'}\n"
                )


def main(args=None):
    rclpy.init(args=args)
    node = ValidationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Results saved to: {node.csv_path}'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
