import numpy as np
from abc import ABC
from pydrake.all import (
        BasicVector,
        Context,
        MultibodyPlant,
        LeafSystem,
        PiecewisePolynomial,
        PiecewisePose,
        RigidTransform,
        Trajectory,
)
# so we can have type annotaitons but prevent circular imports
# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .meta_controller import MetaController
else:
    MetaController = None

from .utils import GripConstants
from .broom_utils import get_broom_grip, get_broom_pregrip, compute_broom_grasp_angle, make_trajectory

# took from pset3, 11_pickplace_initials
def make_trajectory(
    X_Gs: list[RigidTransform], finger_values: np.ndarray, sample_times: list[float]
) -> tuple[Trajectory, PiecewisePolynomial]:
    robot_position_trajectory = PiecewisePose.MakeLinear(sample_times, X_Gs)
    robot_velocity_trajectory = robot_position_trajectory.MakeDerivative()
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)
    return robot_velocity_trajectory, traj_wsg_command

def get_broom_pose(controller: MetaController) -> RigidTransform:
    return controller.plant.EvalBodyPoseInWorld(controller.plant_context, controller.body)

def get_robot_pose(controller: MetaController) -> RigidTransform:
    return controller.plant.EvalBodyPoseInWorld(controller.plant_context, controller.base)

class TrajectoryGenerator(ABC):

    @classmethod
    def trajectory(cls, controller: MetaController) -> tuple[Trajectory, Trajectory]:
        raise NotImplementedError

class PregripToGrip(TrajectoryGenerator):

    trajectory_time = 1

    @classmethod
    def trajectory(cls, controller: MetaController) -> tuple[Trajectory, Trajectory]:
        broom_pose = get_broom_pose(controller)
        robot_pos = get_robot_pose(controller).translation()
        angle = compute_broom_grasp_angle(broom_pose, robot_pos)

        finger_states = np.asarray([GripConstants.opened, GripConstants.opened, GripConstants.closed]).reshape(1, -1)
        angle = compute_broom_grasp_angle(broom_pose, robot_pos)
        pregrip = get_broom_pregrip(broom_pose, angle)
        grip = get_broom_grip(broom_pose, angle)
        gripper_poses = [pregrip, grip, grip]
        times = [0, cls.trajectory_time/2, cls.trajectory_time]
        traj_gripper, traj_wsg = make_trajectory(gripper_poses, finger_states, times)
        return traj_gripper, traj_wsg

class Return(TrajectoryGenerator):
    pass

class MoveToPregrip(TrajectoryGenerator):
    pass

class ManipulateBroom(TrajectoryGenerator):
    pass

