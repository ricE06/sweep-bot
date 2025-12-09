import numpy as np
from abc import ABC
from itertools import pairwise
from pydrake.all import (
        BasicVector,
        Context,
        MultibodyPlant,
        LeafSystem,
        PiecewisePolynomial,
        PiecewisePose,
        RigidTransform,
        RollPitchYaw,
        Trajectory,
)
from scripts.grasp_broom import plan_path
from manipulation.meshcat_utils import AddMeshcatTriad

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
from .ik import solve_ik_for_pose

# took from pset3, 11_pickplace_initials
def make_trajectory(
    X_Gs: list[RigidTransform], finger_values: np.ndarray, sample_times: list[float]
) -> tuple[PiecewisePose, PiecewisePolynomial]:
    robot_position_trajectory = PiecewisePose.MakeLinear(sample_times, X_Gs)
    # robot_velocity_trajectory = robot_position_trajectory.MakeDerivative()
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)
    return robot_position_trajectory, traj_wsg_command

def convert_to_angles(controller: MetaController,
                      pose_trajectory: PiecewisePose,
                      num_divisions: int = 1000):
    # get starting angles of robot
    robot = controller.plant.GetModelInstanceByName('iiwa')
    q_start = controller.plant.GetPositions(controller.plant_context, robot)

    end_time = pose_trajectory.end_time()
    q_frames = [q_start]
    spacing = end_time / num_divisions
    for i in range(num_divisions):
        time = spacing*i
        pose = pose_trajectory.GetPose(time)
        q_next = solve_ik_for_pose(controller.plant, pose, q_frames[-1], pos_tol = 0.03)
        if i % 10 == 0:
            print(q_next)
        q_frames.append(q_next)

    q_frames = np.array(q_frames).T
    # closed on both ends, fencepost
    times = np.linspace(0, end_time, num=num_divisions+1)
    print(times.shape)
    print(len(q_frames))
    return PiecewisePolynomial.FirstOrderHold(times, q_frames)

def get_broom_pose(controller: MetaController) -> RigidTransform:
    return controller.plant.EvalBodyPoseInWorld(controller.plant_context, controller.body)

def get_robot_pose(controller: MetaController) -> RigidTransform:
    return controller.plant.EvalBodyPoseInWorld(controller.plant_context, controller.base)

def get_gripper_pose(controller: MetaController) -> RigidTransform:
    return controller.plant.EvalBodyPoseInWorld(controller.plant_context, controller.gripper)

class TrajectoryGenerator(ABC):

    def trajectory(self, controller: MetaController) -> tuple[Trajectory, Trajectory]:
        raise NotImplementedError

class PregripToGrip(TrajectoryGenerator):

    trajectory_time = 1

    def trajectory(self, controller: MetaController) -> tuple[Trajectory, Trajectory]:
        broom_pose = get_broom_pose(controller)
        print('broom pose', broom_pose)
        robot_pos = get_robot_pose(controller).translation()
        angle = compute_broom_grasp_angle(broom_pose, robot_pos)

        finger_states = np.asarray([GripConstants.opened, GripConstants.opened, GripConstants.closed]).reshape(1, -1)
        angle = compute_broom_grasp_angle(broom_pose, robot_pos)
        pregrip = get_broom_pregrip(broom_pose, angle)
        grip = get_broom_grip(broom_pose, angle)
        gripper_poses = [pregrip, grip, grip]
        times = [0, self.trajectory_time/2, self.trajectory_time]
        traj_gripper, traj_wsg = make_trajectory(gripper_poses, finger_states, times)
        
        """
        end_time = traj_gripper.end_time()
        for i in range(101):
            pose = traj_gripper.GetPose(end_time * i / 100)
            print(pose.translation())
            AddMeshcatTriad(controller.meshcat, f'{i}', X_PT=pose)
        """

        # traj_gripper_q = convert_to_angles(controller, traj_gripper) 
        # return traj_gripper_q, traj_wsg
        # these are poses, will be converted to q and qdot by metacontroller
        return traj_gripper, traj_wsg

class GripToPregrip(TrajectoryGenerator):
    pass

class Return(TrajectoryGenerator):
    trajectory_time = 2

    reset_pose = RigidTransform(RollPitchYaw(np.pi, 0, 0), [0, 1.0, 0.5])

    def trajectory(self, controller: MetaController) -> tuple[Trajectory, Trajectory]:

        current_pose = get_gripper_pose(controller)

        poses = [current_pose, self.reset_pose, self.reset_pose]

        finger_values = np.asarray([GripConstants.closed, GripConstants.opened, GripConstants.opened]).reshape(1, -1)

        times = [0, self.trajectory_time/2, self.trajectory_time]

        traj_gripper, traj_wsg = make_trajectory(poses, finger_values, times)
        return traj_gripper, traj_wsg

class MoveToPregrip(TrajectoryGenerator):

    def trajectory(self, controller: MetaController):
        broom_pose = get_broom_pose(controller)
        robot_pos = get_robot_pose(controller).translation()
        angle = compute_broom_grasp_angle(broom_pose, robot_pos)

        pregrip = get_broom_pregrip(broom_pose, angle)

        X_current = get_gripper_pose(controller)

        traj_gripper, traj_wsg = plan_path(X_current, pregrip, hold_orientation=True, broom_pose=broom_pose)

        return traj_gripper, traj_wsg

target = (0, 0)
reachable_min = (-2, 0)
reachable_max = (2, 2.2)
startable_min = (-2, 2)
startable_max = (2, 2.2)

def get_sweep(controller: MetaController):
    pass

def sweep_to_trajectory(inp: list[tuple[float, float]], broom_start_pose: RigidTransform, robot_center: np.ndarray):
    broom_height = broom_start_pose.translation()[2]
    last_angle = broom_start_pose.rotation().ToRollPitchYaw().yaw_angle()
    grasp_angle = compute_broom_grasp_angle(broom_start_pose, robot_center)
    poses = []
    for a, b in pairwise(inp):
        diff = (b[0] - a[0], b[1] - a[1])
        angle = np.atan2(diff[1], diff[0])
        pose1 = RigidTransform(RollPitchYaw(0, 0, last_angle), np.array([a[0], a[1], broom_height]))
        pose2 = RigidTransform(RollPitchYaw(0, 0, angle), np.array([b[0], b[1], broom_height]))
        last_angle = angle
        poses.extend([get_broom_grip(pose1, grasp_angle), get_broom_grip(pose2, grasp_angle)])
    return poses

class ManipulateBroom(TrajectoryGenerator):

    sweep_generator = SweepGenerator(target, reachable_min, reachable_max, startable_min, startable_max)
    num_point_samples = 200
    time_per_step = 1

    def trajectory(self, controller: MetaController) -> tuple[Trajectory, Trajectory]:
        point_cloud = get_point_cloud(controller)
        robot_center = get_robot_pose(controller).translation()
        broom_pose = get_broom_pose(controller)

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
        poses = sweep_to_trajectory(sweep, broom_pose, robot_center)
        sample_times = [float(i*self.time_per_step) for i in range(len(poses))]
        wsg_poses = np.array([GripConstants.closed]*len(poses)).reshape(1, -1)
        return make_trajectory(poses, wsg_poses, sample_times)
