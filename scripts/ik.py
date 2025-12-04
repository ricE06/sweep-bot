# basically copied from pset 5 oops
# ik solver to go from a gripper target to poses
from pydrake.all import (
    MultibodyPlant,
    RigidTransform,
    InverseKinematics,
    RotationMatrix,
    Solve
)
import numpy as np

def solve_ik_for_pose(
    plant: MultibodyPlant,
    X_WG_target: RigidTransform,
    q_nominal: tuple = tuple(
        np.array([0.0, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0])  # the inital joint poisitions
    ),
    theta_bound: float = 0.01 * np.pi,
    pos_tol: float = 0.015,
) -> tuple:
    """
    Solve IK for a single end-effector pose.

    Args:
        plant: A MultibodyPlant with the iiwa + gripper model.
        X_WG_target: Desired gripper pose in world frame.
        q_nominal: Nominal joint angles for joint-centering.
        theta_bound: Orientation tolerance (radians).
        pos_tol: Position tolerance (meters).

    Returns:
        q_solution: 7 element tuple representing the Optimal
        joint configuration. Each element of the tuple is a float.
    """
    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("body")

    ik = InverseKinematics(plant)
    q_vars = ik.q()[:7]
    prog = ik.prog()

    ik.AddOrientationConstraint(gripper_frame, RotationMatrix(), world_frame, X_WG_target.rotation(), theta_bound)

    ik.AddPositionConstraint(gripper_frame, [0, 0, 0], world_frame, X_WG_target.translation()-[pos_tol]*3, X_WG_target.translation()+[pos_tol]*3)

    prog.AddQuadraticCost(np.eye(len(q_vars)), q_nominal, q_vars)

    prog.SetInitialGuess(q_vars, q_nominal)

    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError("IK did not succeed")

    return tuple(result.GetSolution(q_vars))
