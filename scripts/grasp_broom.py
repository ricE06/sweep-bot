import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    RigidTransform,
    RotationMatrix,
    Parser,
    KinematicTrajectoryOptimization,
    PositionConstraint,
    MinimumDistanceLowerBoundConstraint,
    Solve,
    PiecewisePolynomial,
    RollPitchYaw,
    FixedOffsetFrame,
    OrientationConstraint,

)



def plan_path(X_WStart: RigidTransform, X_WGoal: RigidTransform) -> PiecewisePolynomial:
    """
    Returns joint space trajectory for grasping broom, avoiding collisions between
    iiwa, table, and broom (no gripper or cameras yet)

    """

    # BUILD NEW PLANT WITH EVERYTHING WELDED
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)

    # Add iiwa
    iiwa = parser.AddModelsFromUrl(
        "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf")[0]

    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("iiwa_link_0", iiwa),
        RigidTransform([0, 1.5, 0.0])
    )

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
        plant.world_frame(),
        plant.GetFrameByName("base_link", broom),
        RigidTransform([0.6, 1.2, 0.025])
    )

    # Define gripper frame
    X_gripper = RigidTransform(
        RotationMatrix(RollPitchYaw(np.deg2rad(90), 0, np.deg2rad(90))),
        [0, 0, 0.09]
    )

    gripper_frame = plant.AddFrame(
        FixedOffsetFrame(
            "gripper_frame",
            plant.GetFrameByName("iiwa_link_7", iiwa),
            X_gripper
        ))

    # Finalize plant
    plant.Finalize()
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)

    # ----------------------------------------------------------------------
    # Trajectory optimization
    nq = plant.num_positions()
    print(nq)

    q0 = plant.GetPositions(plant_context)

    trajopt = KinematicTrajectoryOptimization(nq, 10)
    prog = trajopt.get_mutable_prog()

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddDurationConstraint(0.3, 4.0)

    trajopt.AddPositionBounds(
        plant.GetPositionLowerLimits(),
        plant.GetPositionUpperLimits()
    )

    trajopt.AddVelocityBounds(
        plant.GetVelocityLowerLimits(),
        plant.GetVelocityUpperLimits()
    )

    # START constraint
    start_constraint = PositionConstraint(
        plant,
        plant.world_frame(),
        X_WStart.translation(),
        X_WStart.translation(),
        gripper_frame,
        [0, 0, 0],
        plant_context
    )

    trajopt.AddPathPositionConstraint(start_constraint, 0)
    prog.AddQuadraticErrorCost(np.eye(nq), q0, trajopt.control_points()[:, 0])

    # GOAL constraint
    goal_constraint = PositionConstraint(
        plant,
        plant.world_frame(),
        X_WGoal.translation(),
        X_WGoal.translation(),
        gripper_frame,
        [0, 0, 0],
        plant_context
    )

    trajopt.AddPathPositionConstraint(goal_constraint, 1)
    prog.AddQuadraticErrorCost(np.eye(nq), q0, trajopt.control_points()[:, -1])

    # End orientation constraint
    R_WG_goal = RotationMatrix(RollPitchYaw([0, 0, 3.141592]))

    orientation_constraint = OrientationConstraint(
        plant=plant,
        frameAbar=gripper_frame,
        R_AbarA=R_WG_goal,
        frameBbar=plant.world_frame(),
        R_BbarB=RotationMatrix(),
        theta_bound=0.01,
        plant_context=plant_context
    )

    trajopt.AddPathPositionConstraint(orientation_constraint, 1)

    # Start & end velocity = 0
    trajopt.AddPathVelocityConstraint(np.zeros((nq, 1)), np.zeros((nq, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((nq, 1)), np.zeros((nq, 1)), 1)

    # ----------------------------------------------------------------
    # Solve without constraints first

    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError("Initial optimization failed")

    trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))

    # ------------------------------------------------------------
    # Solve with collision constraint
    min_dist_constraint = MinimumDistanceLowerBoundConstraint(plant, 0.05, plant_context, None, 0.1)
    for s in np.linspace(0, 1, 25):
        trajopt.AddPathPositionConstraint(min_dist_constraint, s)

    result = Solve(prog)
    if not result.is_success():
        print("Collision optimization failed")

    return trajopt.ReconstructTrajectory(result)
