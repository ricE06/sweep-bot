import numpy as np
from abc import ABC
from itertools import pairwise
from pydrake.all import (
        BasicVector,
        Context,
        Meshcat,
        MultibodyPlant,
        LeafSystem,
        PiecewisePolynomial,
        PiecewisePose,
        RigidTransform,
        RollPitchYaw,
        RotationMatrix,
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

    def trajectory(self, controller: MetaController) -> Trajectory:
        raise NotImplementedError

class RotateGrip(TrajectoryGenerator):
    # intiialize with goal pose of broom

    trajectory_time = 2.0
    num_steps = 10

    def __init__(self, broom_goal: RigidTransform):
        self.broom_goal = broom_goal

    def trajectory(self, controller: MetaController):
        broom_current = get_broom_pose(controller)
        gripper_current = get_gripper_pose(controller)


        # linearly interpolate 10 different broom rotations in between
        sample_times = [0.0, self.trajectory_time]
        X_Gs = [broom_current, self.broom_goal]

        broom_traj = PiecewisePose.MakeLinear(sample_times, X_Gs)

        # sample uniformly
        times = np.linspace(0, self.trajectory_time, self.num_steps + 1)
        broom_poses = [broom_traj.get_pose(t) for t in times]

        # gripper to broom base
        X_GB = gripper_current.inverse().multiply(broom_current)

        gripper_poses = []
        for X_WB in broom_poses:
            X_WG = X_WB.multiply(X_GB.inverse())
            gripper_poses.append(X_WG)

        finger_values = np.array([[GripConstants.closed] * len(gripper_poses)])

        times = np.linspace(0, self.trajectory_time, len(gripper_poses)).tolist()

        traj_gripper, traj_wsg = make_trajectory(gripper_poses, finger_values, times)
        return traj_gripper, traj_wsg

class PregripToGrip(TrajectoryGenerator):

    trajectory_time = 1

    def trajectory(self, controller: MetaController) -> tuple[PiecewisePose, Trajectory]:
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

    def trajectory(self, controller: MetaController) -> [Trajectory, Trajectory]:

        current_pose = get_gripper_pose(controller)

        poses = [current_pose, self.reset_pose, self.reset_pose]

        finger_values = np.asarray([GripConstants.closed, GripConstants.opened, GripConstants.opened]).reshape(1, -1)

        times = [0, self.trajectory_time/2, self.trajectory_time]

        traj_gripper, traj_wsg = make_trajectory(poses, finger_values, times)
        return traj_gripper, traj_wsg

class MoveToStart(TrajectoryGenerator):

    def __init__(self, meshcat = None):
        self.meshcat = meshcat

    num_retries = 1
    start_pose = RigidTransform(RotationMatrix(), np.array([0, 0.8, 0.025]))

    def trajectory(self, controller: MetaController) -> Trajectory:
        broom_pose = get_broom_pose(controller)
        # X_current = get_gripper_pose(controller)

        success = False
        attempts = 0
        while not success and attempts < self.num_retries:
            if self.meshcat:
                AddMeshcatTriad(self.meshcat, f'trajopt-{attempts}', X_PT=self.start_pose)
            traj_gripper, traj_wsg, success = plan_path(broom_pose, self.start_pose, hold_orientation=True) 
            attempts += 1

        print('attempts needed:', attempts)

        return traj_gripper

target = (0, 0)
reachable_min = (-2, 0)
reachable_max = (2, 2.2)
startable_min = (-0.6, 0.8)
startable_max = (0.6, 0.81)

def get_sweep(controller: MetaController):
    pass

def sweep_to_trajectory(inp: list[tuple[float, float]], broom_start_pose: RigidTransform):
    # broom_height = broom_start_pose.translation()[2]
    broom_height = 0.025
    last_angle = broom_start_pose.rotation().ToRollPitchYaw().yaw_angle()
    poses = []
    for a, b in pairwise(inp):
        diff = (b[0] - a[0], b[1] - a[1])
        angle = np.atan2(diff[1], diff[0])
        pose1 = RigidTransform(RollPitchYaw(0, 0, last_angle+np.pi), np.array([a[0], a[1], broom_height]))
        pose2 = RigidTransform(RollPitchYaw(0, 0, angle+np.pi), np.array([b[0], b[1], broom_height]))
        last_angle = angle
        poses.extend([pose1, pose2])
    return poses

class ManipulateBroom(TrajectoryGenerator):

    def __init__(self, meshcat: Meshcat | None = None):
        self.meshcat = meshcat

    sweep_generator = SweepGenerator(target, reachable_min, reachable_max, startable_min, startable_max)
    num_point_samples = 200
    time_per_step = 1

    def trajectory(self, controller: MetaController) -> Trajectory:
        point_cloud = get_point_cloud(controller)
        # robot_center = get_robot_pose(controller).translation()
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
        poses = sweep_to_trajectory(sweep, broom_pose) 
        if self.meshcat:
            for i, pose in enumerate(poses):
                AddMeshcatTriad(self.meshcat, f'sweep_{i}', X_PT = pose)

        sample_times = [float(i*self.time_per_step) for i in range(len(poses))]
        return PiecewisePose.MakeLinear(sample_times, poses)
        return make_trajectory(poses, wsg_poses, sample_times)
