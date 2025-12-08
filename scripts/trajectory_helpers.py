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
from .broom_utils import get_broom_grip, get_broom_pregrip, compute_broom_grasp_angle 
from .point_cloud import get_point_cloud
from .sample_sweep import SweepGenerator

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

    def trajectory(self, controller: MetaController) -> tuple[Trajectory, Trajectory]:
        raise NotImplementedError

class PregripToGrip(TrajectoryGenerator):

    trajectory_time = 1

    def trajectory(self, controller: MetaController) -> tuple[Trajectory, Trajectory]:
        broom_pose = get_broom_pose(controller)
        robot_pos = get_robot_pose(controller).translation()
        angle = compute_broom_grasp_angle(broom_pose, robot_pos)

        finger_states = np.asarray([GripConstants.opened, GripConstants.opened, GripConstants.closed]).reshape(1, -1)
        angle = compute_broom_grasp_angle(broom_pose, robot_pos)
        pregrip = get_broom_pregrip(broom_pose, angle)
        grip = get_broom_grip(broom_pose, angle)
        gripper_poses = [pregrip, grip, grip]
        times = [0, self.trajectory_time/2, self.trajectory_time]
        traj_gripper, traj_wsg = make_trajectory(gripper_poses, finger_states, times)
        return traj_gripper, traj_wsg

class Return(TrajectoryGenerator):
    pass

class MoveToPregrip(TrajectoryGenerator):
    pass

target = (0, 0)
reachable_min = (-2, 0)
reachable_max = (2, 2.2)
startable_min = (-2, 2)
startable_max = (2, 2.2)
class ManipulateBroom(TrajectoryGenerator):

    sweep_generator = SweepGenerator(target, reachable_min, reachable_max, startable_min, startable_max)
    num_point_samples = 200

    def trajectory(self, controller: MetaController) -> tuple[Trajectory, Trajectory]:
        point_cloud = get_point_cloud(controller)

        point_cloud_xy = point_cloud[0:2, :]
        num_points = len(point_cloud_xy)
        if num_points > self.num_point_samples:
            sample_idxs = np.random.choice(np.arange(num_points), self.num_point_samples, replace=False)
        else:
            sample_idxs = np.arange(num_points)
        point_set = set()
        for i in sample_idxs:
            point_set.add((point_cloud_xy[i, 0], point_cloud_xy[i, 1]))

        sweep = self.sweep_generator.find_sweep(point_set)
        print(sweep)

traj = [PregripToGrip, Return, MoveToPregrip, ]
traj[2].trajectory()

