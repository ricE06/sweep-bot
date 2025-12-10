import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    RigidTransform,
    RotationMatrix,
    Parser,
    KinematicTrajectoryOptimization,
    PositionConstraint,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MinimumDistanceLowerBoundConstraint,
    Solve,
    PiecewisePolynomial,
    PiecewisePose,
    Role,
    RollPitchYaw,
    FixedOffsetFrame,
    OrientationConstraint,
    PathParameterizedTrajectory,
    CompositeTrajectory,
    BsplineTrajectory,
    Trajectory,
)
from manipulation.scenarios import AddIiwa, AddWsg
from .ik import solve_ik_for_pose
from .utils import GripConstants


def build_temp_plant(q0 = None, meshcat = None):

    # BUILD NEW PLANT WITH EVERYTHING WELDED
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)

    # Add iiwa
    # iiwa = parser.AddModelsFromUrl(
    #     "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf")[0]
    iiwa = AddIiwa(plant, collision_model="with_box_collision")

    joint_name = 'world_welds_to_iiwa_link_0'
    if plant.HasJointNamed(joint_name):
        joint = plant.GetJointByName(joint_name)
        plant.RemoveJoint(joint)
    else:
        print(f"Joint '{joint_name}' not found.")

    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("iiwa_link_0", iiwa),
        RigidTransform(RollPitchYaw(0, 0, 0), [0, 1.6, 0.0])
    )

    wsg = AddWsg(plant, iiwa, welded=True, sphere=True)

    # Add table
    table = parser.AddModels("./models/table_with_hole.sdf")[0]

    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("hole_floor", table),
        RigidTransform([0.0, 0.0, -0.55])
    )

    # Add broom
    broom = parser.AddModels("./models/broom.sdf")[0]


    plant.WeldFrames(
        plant.GetFrameByName("body", wsg),
        plant.GetFrameByName("handle_link", broom),
        RigidTransform(RollPitchYaw(np.pi, 0, 0), np.array([0, 0.35, 0.52])),
    )

    broom_frame = plant.GetFrameByName("handle_link", broom)

    # Finalize plant
    plant.Finalize()
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    if q0 is None:
        q0 = plant.GetPositions(plant_context)
    # plant.SetPositions(plant_context, iiwa, q0)

    return diagram, plant, broom_frame


def plan_path(X_WStart: RigidTransform, X_WGoal: RigidTransform,
              q0 = None,
              hold_orientation: bool = False) -> tuple[Trajectory, Trajectory, bool]:
    """
    Returns joint space trajectory for grasping broom, avoiding collisions between
    iiwa, table, and broom (no gripper or cameras yet)
    """

    diagram, plant, broom_frame = build_temp_plant(q0)
    print(plant)
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)

    if q0 is None:
        q0 = plant.GetPositions(plant_context)


    # ----------------------------------------------------------------------
    # Trajectory optimization
    nq = plant.num_positions()
    print(nq)

    trajopt = KinematicTrajectoryOptimization(nq, 10)
    prog = trajopt.get_mutable_prog()

    # start_pose = solve_ik_for_pose(plant, X_WStart)
    # goal_pose = solve_ik_for_pose(plant, X_WGoal)
    q_guess = np.tile(q0.reshape((7, 1)), (1, trajopt.num_control_points()))
    print(q_guess.shape)
    q_guess[0, :] = np.linspace(0, np.pi/2, trajopt.num_control_points())
    # print(q_guess.shape)
    # print(trajopt.num_control_points())
    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
    trajopt.SetInitialGuess(path_guess)


    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddDurationConstraint(0.1, 3.5)

    trajopt.AddPositionBounds(
        plant.GetPositionLowerLimits(),
        plant.GetPositionUpperLimits()
    )

    trajopt.AddVelocityBounds(
        plant.GetVelocityLowerLimits(),
        plant.GetVelocityUpperLimits()
    )

    pos_tol = np.array([0.01, 0.01, 0.01])
    pos_tol = np.array([0, 0, 0])

    # START constraint
    start_constraint = PositionConstraint(
        plant,
        plant.world_frame(),
        X_WStart.translation() - pos_tol,
        X_WStart.translation() + pos_tol,
        broom_frame,
        [0, 0, 0],
        plant_context
    )

    trajopt.AddPathPositionConstraint(start_constraint, 0)
    prog.AddQuadraticErrorCost(np.eye(nq), q0, trajopt.control_points()[:, 0])

    # GOAL constraint
    goal_constraint = PositionConstraint(
        plant,
        plant.world_frame(),
        X_WGoal.translation() - pos_tol,
        X_WGoal.translation() + pos_tol,
        broom_frame,
        [0, 0, 0],
        plant_context
    )

    trajopt.AddPathPositionConstraint(goal_constraint, 1)
    prog.AddQuadraticErrorCost(np.eye(nq), q0, trajopt.control_points()[:, -1])

    # End orientation constraint
    R_WG_goal = X_WGoal.rotation()

    orientation_constraint = OrientationConstraint(
        plant=plant,
        frameAbar=broom_frame,
        R_AbarA=RotationMatrix(),
        frameBbar=plant.world_frame(),
        R_BbarB=R_WG_goal,
        theta_bound=0.001,
        plant_context=plant_context
    )

    trajopt.AddPathPositionConstraint(orientation_constraint, 1)

    # Start & end velocity = 0
    trajopt.AddPathVelocityConstraint(np.zeros((nq, 1)), np.zeros((nq, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((nq, 1)), np.zeros((nq, 1)), 1)

    # ----------------------------------------------------------------
    # Solve without constraints first

    # result = Solve(prog)
    # if not result.is_success():
    #     raise RuntimeError("Initial optimization failed")

    # trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))

    # ------------------------------------------------------------
    # Solve with collision constraint
    min_dist_constraint = MinimumDistanceLowerBoundConstraint(plant, 0.001, plant_context, None, 1e-6)
    for s in np.linspace(0, 1, 25):
        trajopt.AddPathPositionConstraint(min_dist_constraint, s)

    result = Solve(prog)
    if not result.is_success():
        print('Solver failed')
        # print(result.get_solver_details().__dict__)
        print(result.GetInfeasibleConstraintNames(prog))

    traj_V_G: Trajectory = trajopt.ReconstructTrajectory(result)
    sample_times = [0, traj_V_G.end_time()]
    finger_values = np.array([GripConstants.opened, GripConstants.opened]).reshape(1, -1)
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)

    return traj_V_G, traj_wsg_command, result.is_success()

def straight_path(plant, q_current, X_WGoal):
    """
    Create a simple straight line joint space trajectory from current joint position to
    goal assuming both IK solutions are reachable.

    Args:
        q_current: current joint position of iiwa
        X_WGoal: the goal location in workspace
    """

    # Get IK solutions
    q_current = q_current.reshape(-1)
    q_goal = solve_ik_for_pose(plant, X_WGoal, q_nominal=q_current)

    q_mat = np.vstack([q_current, q_goal]).T  # shape (nq, 2)
    times = [0, 0.75]  # 0.75 seconds approach

    traj = PiecewisePolynomial.FirstOrderHold(times, q_mat)
    return traj


def grasp_path(X_WStart, X_WPregrasp, X_WGrasp, X_WLift):
    opened = GripConstants.opened
    closed = GripConstants.closed

    # start -> pregrasp
    arm_1, _ = plan_path(X_WStart, X_WPregrasp)
    q_pre = arm_1.value(arm_1.end_time())

    # approach: pregrasp -> grasp (straight line)
    diagram, plant, gripper_frame = build_temp_plant()
    arm_2 = straight_path(plant, q_pre, X_WGrasp)
    q_grasp = arm_2.value(arm_2.end_time())

    # lift: grasp -> lift
    arm_3 = straight_path(plant, q_grasp, X_WLift)

    # combine
    arm_traj = CompositeTrajectory.AlignAndConcatenate([arm_1, arm_2, arm_3])

    # get end times
    t1 = arm_1.end_time()
    t2 = t1 + arm_2.end_time()
    t3 = arm_traj.end_time()

    # build gripper trajectory
    wsg_times = [0, t1, t2, t2 + 0.2, t3]
    wsg_pos   = np.array([[opened, opened, opened, closed, closed]])

    wsg_traj = PiecewisePolynomial.FirstOrderHold(wsg_times, wsg_pos)

    return arm_traj, wsg_traj
