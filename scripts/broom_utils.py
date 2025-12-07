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
PREGRIP_DIST = 0.2

# helper functions to compute broom related poses

# currently broken, should try to point to center of robot
def compute_broom_grasp_angle(broom_base: RigidTransform, robot_center: np.ndarray):
    X_Whandle: RigidTransform = broom_base @ RigidTransform(RotationMatrix(), X_offset.translation())
    handle_pos = X_Whandle.translation()
    robot_to_handle = handle_pos - robot_center
    angle = np.atan2(robot_to_handle[1], robot_to_handle[0]) - np.pi/2
    return angle

def compute_wrist_rotation_angle(cur_grasp: RigidTransform): 
    # project out component in direction of the gripper
    gripper_vec = cur_grasp.rotation().matrix()[:, 1]
    print(gripper_vec)

def _get_broom_offset(broom_base: RigidTransform, angle: float, offset: float):
    """
    Computes a grip for the broom at broom_base (on the handle), pointing in angle,
    with offset (positive radially outward at this angle)
    """
    broom_transposed = broom_base @ RigidTransform(RollPitchYaw(0, 0, angle), X_offset.translation())
    return broom_transposed

def get_broom_grip(broom_base: RigidTransform, angle: float):
    return _get_broom_offset(broom_base, angle, GRIP_DIST)

def get_broom_pregrip(broom_base: RigidTransform, angle: float):
    # move to the handle location, but don't rotate
    X_Whandle: RigidTransform = broom_base @ X_Joint @ X_handle_offset
    X_Whandle_rotated = X_Whandle @ RigidTransform(RollPitchYaw(0, 0, -angle), np.array([0, 0, 0])) 
    # X_Whandle: RigidTransform = broom_base @ RigidTransform(RotationMatrix(), X_Joint.translation()) @ X_handle_offset
    X_Whandle_rotated = X_Whandle @ RigidTransform(RollPitchYaw(0, 0, -angle), np.array([0, 0, 0])) 
    X_out =  X_Whandle_rotated @ RigidTransform(RollPitchYaw(0, np.pi, 0), np.array([0, -PREGRIP_DIST, 0]))
    return X_out
