import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Context,
    Diagram,
    DiagramBuilder,
    RigidTransform,
    RotationMatrix,
    Parser,
    KinematicTrajectoryOptimization,
    PositionConstraint,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MinimumDistanceLowerBoundConstraint,
    MultibodyPlant,
    Solve,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
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


def plan_sweep(plant: MultibodyPlant, plant_context: Context,
               broom_frame,
        X_WStart: RigidTransform, X_WGoal: RigidTransform,) -> tuple[Trajectory, Trajectory, bool]:
    """
    Returns joint space trajectory for grasping broom, avoiding collisions between
    iiwa, table, and broom (no gripper or cameras yet)
    """

    if q0 is None:
        q0 = plant.GetPositions(plant_context)

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

    height_constraint = PositionConstraint(
        plant,
        frameA=plant.world_frame(),
        p_A_lower=np.array([-np.inf, -np.inf, -np.inf]),
        p_A_upper=np.array([ np.inf,  np.inf, 0.05]),
        frameB=broom_frame,
        p_B=np.array([0, 0, 0]),
        plant_context=plant_context
    )
    for s in np.linspace(0, 1, 20):
        trajopt.AddPathPositionConstraint(height_constraint, s)

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
