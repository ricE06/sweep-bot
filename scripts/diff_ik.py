import numpy as np
from pydrake.all import (
        BasicVector,
        Context,
        JacobianWrtVariable,
        LeafSystem,
        MultibodyPlant,
)

class PseudoInverseController(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context: Context, output: BasicVector):
        """
        fill in our code below.
        """
        # evaluate the V_G_port and q_port on the current context to get those values.
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)

        self._plant.SetPositions(self._plant_context, self._iiwa, q)

        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kQDot, self._G, [0,0,0], self._W, self._W
        )[:, self.iiwa_start : self.iiwa_end + 1]

        # compute `v` by mapping the gripper velocity (from the V_G_port) to the joint space
        v = np.linalg.pinv(J_G) @ V_G
        output.SetFromVector(v)
