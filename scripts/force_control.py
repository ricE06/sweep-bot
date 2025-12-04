import numpy as np
from pydrake.all import (
    RigidTransform, RotationMatrix,
    JacobianWrtVariable, SpatialForce
)

# broom bottom friction low
# high friction for stick, gripper
def hybrid_force_position_control(
        plant, context,
        frame_E,                              # broom/end-effector frame
        x_des,                                 # desired SE(3) pose RigidTransform
        Fz_des=-5.0,                            # desired downward force (N, negative = down)
        Kp_pos=np.array([200, 200, 0]),         # XYZ position gains (0 for z → force controlled)
        Kd_pos=np.array([20, 20, 0]),           # XYZ velocity gains
        Kp_force=5.0,                           # scalar gain for force control in z
):
    """
    Implements hybrid force/position control:
    - X,Y = position control
    - Z = force control

    Returns:
        tau : np.array[nq] joint torque command
    """

    X_WE = plant.CalcRelativeTransform(context, plant.world_frame(), frame_E)
    v_WE = plant.CalcSpatialVelocity(
        context,
        frame_E,
        plant.world_frame(),                     # velocity expressed in world
        plant.world_frame()
    )

    p_current = X_WE.translation()
    p_des     = x_des.translation()

    pos_err = np.array([
        p_des[0] - p_current[0],
        p_des[1] - p_current[1],
        0.0   #foce controlled
    ])

    v_lin = v_WE.translational()
    # pos control
    F_pos = Kp_pos * pos_err - Kd_pos * v_lin
    # force control
    measured_F_E = np.zeros(3)   # add sensor here

    Fz_err = Fz_des - measured_F_E[2]

    F_force = np.array([0., 0., Kp_force * Fz_err])

    # Desired spatial force in world frame
    F_total_world = SpatialForce(
        rotational=np.zeros(3),
        translational=F_pos + F_force
    )

    #spatial force to joint torques
    Jv = plant.CalcJacobianTranslationalVelocity(
        context,
        JacobianWrtVariable.kQDot,
        frame_E,
        np.zeros(3),                               # point at origin of EE frame
        plant.world_frame(),
        plant.world_frame()
    )

    tau = Jv.T @ F_total_world.translational()

    return tau

#implementation should look something like this

while running:
    context = diagram.GetMutableSubsystemContext(plant, root_context)

    x_des = RigidTransform(
        RotationMatrix.MakeZRotation(0),
        np.array([x_goal, y_goal, z_contact_height])
    )

    tau = hybrid_force_position_control(
        plant, context,
        frame_E=broom_frame,
        x_des=x_des,
        Fz_des=-8.0    # push with 8 N downward
    )

    command_sender.SendTorque(tau)
