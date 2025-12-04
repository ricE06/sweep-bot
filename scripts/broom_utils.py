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

GRIP_DIST = 0.01
PREGRIP_DIST = 0.2

# helper functions to compute broom related poses

# currently broken, should try to point to center of robot
def compute_broom_grasp_angle(broom_base: RigidTransform, robot_center: np.ndarray):
    X_Whandle: RigidTransform = broom_base @ X_Joint @ X_handle_offset
    handle_pos = X_Whandle.translation()

    # compute amount of rotation need to align to robot center
    y_vec = X_Whandle.rotation().matrix() @ np.array([0, 1, 0])
    z_vec = X_Whandle.rotation().matrix() @ np.array([0, 0, 1])
    robot_to_handle = handle_pos - robot_center
    print(robot_to_handle)
    robot_to_handle_proj = robot_to_handle - (np.dot(robot_to_handle, z_vec) * z_vec / np.linalg.norm(z_vec))
    angle = angle_between_vectors(y_vec, robot_to_handle_proj)

    # do we turn left or right? need dot prod with orthogonal y to know
    # sign_bool = angle_between_vectors(np.array([-y_vec[1], y_vec[0]]), robot_to_handle[0:2]) >= 0
    # sign = 1 if sign_bool else -1
    # angle = -sign * angle
    print(angle)
    return angle


def get_broom_grip(broom_base: RigidTransform, angle: float):
    X_Whandle: RigidTransform = broom_base @ X_Joint @ X_handle_offset
    X_Whandle_rotated = X_Whandle @ RigidTransform(RollPitchYaw(0, 0, -angle), np.array([0, 0, 0])) 
    X_out =  X_Whandle_rotated @ RigidTransform(RollPitchYaw(0, np.pi, 0), np.array([0, -GRIP_DIST, 0]))

    return X_out

def get_broom_pregrip(broom_base: RigidTransform, angle: float):
    # move to the handle location, but don't rotate
    X_Whandle: RigidTransform = broom_base @ X_Joint @ X_handle_offset
    X_Whandle_rotated = X_Whandle @ RigidTransform(RollPitchYaw(0, 0, -angle), np.array([0, 0, 0])) 
    # X_Whandle: RigidTransform = broom_base @ RigidTransform(RotationMatrix(), X_Joint.translation()) @ X_handle_offset
    X_Whandle_rotated = X_Whandle @ RigidTransform(RollPitchYaw(0, 0, -angle), np.array([0, 0, 0])) 
    X_out =  X_Whandle_rotated @ RigidTransform(RollPitchYaw(0, np.pi, 0), np.array([0, -PREGRIP_DIST, 0]))
    return X_out
