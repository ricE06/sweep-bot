import numpy as np
from pydrake.all import (
        RigidTransform,
        RollPitchYaw,
        RotationMatrix,
)

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))

# from broom sdf
X_Joint = RigidTransform(RollPitchYaw(-0.5236, 0, 0), np.array([0, 0.2, 0.35]))
X_handle_offset = RigidTransform(RollPitchYaw(0, 0, 0), np.array([0, 0, 0.2]))

X_offset = X_Joint @ X_handle_offset

GRIP_DIST = 0.01
PREGRIP_DIST = 0.18

# helper functions to compute broom related poses

# currently broken, should try to point to center of robot
def compute_broom_grasp_angle(broom_base: RigidTransform, robot_center: np.ndarray):
    X_Whandle: RigidTransform = broom_base @ RigidTransform(RotationMatrix(), X_offset.translation())
    handle_pos = X_Whandle.translation()
    robot_to_handle = handle_pos - robot_center
    angle = np.atan2(robot_to_handle[1], robot_to_handle[0]) - np.pi/2
    return angle

def compute_wrist_rotation_angle(cur_grasp: RigidTransform): 
    # we need to make gripper x orthogonal to broom dir
    gripper_x = cur_grasp.rotation().matrix()[:, 0]
    broom_direction = cur_grasp.InvertAndCompose(X_offset)
    roll_ang = broom_direction.rotation().ToRollPitchYaw().roll_angle()
    return roll_ang

def _get_broom_offset(broom_base: RigidTransform, angle: float, offset: float):
    """
    Computes a grip for the broom at broom_base (on the handle), pointing in angle,
    with offset (positive radially outward at this angle)
    """
    broom_transposed = broom_base @ RigidTransform(RollPitchYaw(0, 0, angle), X_offset.translation())
    wrist_angle = compute_wrist_rotation_angle(broom_transposed)
    broom_grip = broom_transposed @ RigidTransform(RollPitchYaw(wrist_angle, 0, 0), np.array([0, 0, 0]))
    broom_grip_offset = broom_grip @ RigidTransform(RotationMatrix(), np.array([0, -offset, 0]))
    return broom_grip_offset

def get_broom_grip(broom_base: RigidTransform, angle: float):
    return _get_broom_offset(broom_base, angle, GRIP_DIST)

def get_broom_pregrip(broom_base: RigidTransform, angle: float):
    return _get_broom_offset(broom_base, angle, PREGRIP_DIST)
