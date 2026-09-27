#!/usr/bin/env python3
"""
pixel_to_world.py
==================
Purpose:
    Convert a 2D pixel coordinate (u, v) from the camera image into a
    3D world coordinate (X, Y, Z) on the table surface.

How it works (simple version):
    1. From the pixel, build a ray in camera frame.
    2. Rotate that ray into world frame using camera's orientation.
    3. Intersect the ray with the flat table plane (Z = table_top_z).
    4. Return the intersection point in world coordinates.

Important convention notes (Isaac Sim specific):
    - Isaac Sim uses XYZ Euler order in the Transform panel:
        R = Rx @ Ry @ Rz
    - Isaac Sim cameras look along -Z of their LOCAL frame
      (NOT +Z like OpenCV).
    - Isaac Sim camera local frame: +X right, +Y up, -Z forward.
    - Pixel (u, v): u=0 left, v=0 top (standard image convention).

How it's used:
    converter = PixelToWorldConverter(config_dict)
    world_xyz = converter.pixel_to_world(pixel_u=320, pixel_v=400)

Why this approach:
    - Works for any static scene where objects sit on a known flat plane.
    - Requires no depth camera (RGB only).
    - Uses the same math in simulation AND real world — only the config
      file values change.

Author: Master's Thesis — LLM-Coordinated Pfand Sorting System
"""

import math
import numpy as np
from typing import Tuple, Optional


class PixelToWorldConverter:
    """
    Converts pixel coordinates to world coordinates using the known
    camera pose and the assumption that objects rest on a flat table.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dictionary loaded from camera_config.yaml.
        """
        self.config = config

        # ----- Intrinsics -----
        cam_cfg = config['camera']
        self.image_w = cam_cfg['image_width']
        self.image_h = cam_cfg['image_height']
        self.focal_mm = cam_cfg['focal_length_mm']
        self.h_aperture_mm = cam_cfg['horizontal_aperture_mm']
        self.v_aperture_mm = cam_cfg['vertical_aperture_mm']

        # Pixel focal lengths and principal point
        self.fx = self.focal_mm * self.image_w / self.h_aperture_mm
        self.fy = self.focal_mm * self.image_h / self.v_aperture_mm
        self.cx = self.image_w / 2.0
        self.cy = self.image_h / 2.0

        # ----- Extrinsics -----
        pos = cam_cfg['position']
        self.cam_pos = np.array([pos['x'], pos['y'], pos['z']],
                                dtype=np.float64)

        rot = cam_cfg['rotation_euler_deg']
        self.cam_rot_deg = (rot['x'], rot['y'], rot['z'])
        # Isaac Sim Transform panel uses XYZ intrinsic order.
        self.R_cam_to_world = self._euler_xyz_to_matrix(
            rot['x'], rot['y'], rot['z']
        )

        # ----- Table plane -----
        self.table_z = config['table']['top_z']

        # ----- Robot -----
        rb = config['robot']['base_position']
        self.robot_base = np.array([rb['x'], rb['y'], rb['z']],
                                   dtype=np.float64)
        self.robot_max_reach = config['robot']['max_reach_m']

    # ------------------------------------------------------------------
    # Core conversion
    # ------------------------------------------------------------------

    def pixel_to_world(
        self,
        pixel_u: float,
        pixel_v: float,
    ) -> Optional[np.ndarray]:
        """
        Convert a pixel (u, v) to world coordinates (X, Y, Z) on the
        table plane.
        """
        # Step 1: Ray in CAMERA frame.
        # Isaac Sim camera local frame: +X right, +Y up, -Z forward.
        # Image: u increases rightward (same as +X),
        #        v increases downward (OPPOSITE to +Y).
        x_cam = (pixel_u - self.cx) / self.fx
        y_cam = -(pixel_v - self.cy) / self.fy   # flip: image-down vs cam-up
        z_cam = -1.0                             # camera looks along -Z
        ray_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

        # Step 2: Rotate into WORLD frame.
        ray_world = self.R_cam_to_world @ ray_cam
        norm = np.linalg.norm(ray_world)
        if norm < 1e-12:
            return None
        ray_world = ray_world / norm

        # Step 3: Intersect with table plane Z = table_z.
        if abs(ray_world[2]) < 1e-8:
            return None
        t = (self.table_z - self.cam_pos[2]) / ray_world[2]
        if t <= 0:
            return None
        return self.cam_pos + t * ray_world

    # ------------------------------------------------------------------
    # Bounding-box helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pick_grounding_pixel(
        x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[float, float, str]:
        """
        Choose the pixel that represents where the object touches the
        table, plus a pose label.

        Both poses are grounded at the BOTTOM-CENTRE of the bounding box
        (center_x, y2); only the pose label differs:
          - Tall box (height > width)  -> STANDING
          - Wide box (width >= height) -> LYING
        """
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2.0
        if height > width:
            return center_x, y2, "standing"
        return center_x, y2, "lying"

    # ------------------------------------------------------------------
    # Frame transforms & reachability
    # ------------------------------------------------------------------

    def world_to_robot_base(self, world_point: np.ndarray) -> np.ndarray:
        """Convert a world point to robot-base-relative coordinates."""
        return world_point - self.robot_base

    def distance_from_robot_base(self, world_point: np.ndarray) -> float:
        """Horizontal distance from robot base (X, Y only)."""
        dx = world_point[0] - self.robot_base[0]
        dy = world_point[1] - self.robot_base[1]
        return math.sqrt(dx * dx + dy * dy)

    def is_reachable(self, world_point: np.ndarray) -> bool:
        """Is this world point within the robot's reach?"""
        return self.distance_from_robot_base(world_point) \
            <= self.robot_max_reach

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _euler_xyz_to_matrix(
        rx_deg: float, ry_deg: float, rz_deg: float
    ) -> np.ndarray:
        """
        Rotation matrix from Euler XYZ intrinsic: R = Rx @ Ry @ Rz.
        Matches Isaac Sim's Transform panel behavior.
        """
        rx = math.radians(rx_deg)
        ry = math.radians(ry_deg)
        rz = math.radians(rz_deg)

        Rx = np.array([
            [1,           0,            0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx),  math.cos(rx)],
        ])
        Ry = np.array([
            [ math.cos(ry), 0, math.sin(ry)],
            [            0, 1,            0],
            [-math.sin(ry), 0, math.cos(ry)],
        ])
        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz),  math.cos(rz), 0],
            [           0,             0, 1],
        ])
        return Rx @ Ry @ Rz
