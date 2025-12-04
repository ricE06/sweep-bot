import numpy as np
from pydrake.all import (
        BasicVector,
        Context,
        MultibodyPlant,
        LeafSystem,
)

# THE FOLLOWING IS LEFT TO DO:
#
# Metacontroller oversees a several step process:
# First, it outputs positions from the trajectory opt.
# Then, it outputs positions from a TODO defined function that takes in
# the pregrip pose and the grip pose, runs a make_interpolant_trajectory, outputs that
# Then, it outputs positions from a TODO defined function that takes in
# a bunch of broom base poses and returns grip poses.
# When using the broom grasp calculator, set the angle to 0 for now.
# You likely need a specialized "switch" input port to feed signals into
# in order to determine which of these phases we are in.

# The make_interpolant_trajectory will take in keyframes (similar to pset)
# of desired positions + the width of the gripper. It will go through them
# one by one for some (configurable) time frame.
# If any of you figure out how to dynamically string together trajectory sources,
# that is amazing and we should do that, but my intuition leads me to
# writing a custom controller for this
# Essentially just a giant switch case statement

# (after mtg)
# turn on position + torque control. ideally we'd be in full torque control
# once we have a stiffness controller for the first phase
# implement a phase that moves the broom back after a sweep
# call the point clouds sampling to determine the broom base poses

class MetaController(LeafSystem):
    """
    Controller that provides commands to the iiwa
    depending on where in the simulation we are in.
    """

    def __init__(self, plant: MultibodyPlant,
                 traj_opt,
                 traj_pregrip_to_grip,
                 traj_sweep,
                 traj_return=None):
        super().__init__()

        self._plant = plant
        self._nq = plant.num_positions()
        self._traj_opt = traj_opt
        self._traj_pregrip = traj_pregrip_to_grip
        self._traj_sweep = traj_sweep
        self._traj_return = traj_return

        # compute durations for stitch
        self._t_opt = traj_opt.end_time() - traj_opt.start_time()
        self._t_pregrip = traj_pregrip_to_grip.end_time() - traj_pregrip_to_grip.start_time()
        self._t_sweep = traj_sweep.end_time() - traj_sweep.start_time()
        self._t_return = (traj_return.end_time() - traj_return.start_time()) if traj_return else 0

        #cummulative phase times
        self._phase_times = [
            self._t_opt,
            self._t_opt + self._t_pregrip,
            self._t_opt + self._t_pregrip + self._t_sweep,
            self._t_opt + self._t_pregrip + self._t_sweep + self._t_return,
        ]

        self.DeclareVectorOutputPort(
            "position_command",
            BasicVector(self._nq),
            self.CalcPosOutput
        )
    def _get_phase_and_local_time(self, t):
        """
        Given global simulation time t, determine:
        - current phase index (0, 1, 2, or 3)
        - local time within that trajectory
        """
        # Phase 0: traj_opt
        if t < self._phase_times[0]:
            return 0, t
        # Phase 1: pregrip → grip
        if t < self._phase_times[1]:
            return 1, t - self._phase_times[0]
        # phase 2: sweeping
        if t < self._phase_times[2]:
            return 2, t - self._phase_times[1]
        # Phase 3: return
        if self._traj_return:
            local_t = min(t - self._phase_times[2], self._t_return)
            return 3, local_t
        # No return trajectory, maintain last position
        return 2, self._t_sweep

    def CalcPosOutput(self, context: Context, output: BasicVector):
        t = context.get_time()
        phase, local_t = self._get_phase_and_local_time(t)
        if phase == 0:
            traj = self._traj_opt
        elif phase == 1:
            traj = self._traj_pregrip
        elif phase == 2:
            traj = self._traj_sweep
        else:
            traj = self._traj_return

        local_t = np.clip(local_t, traj.start_time(), traj.end_time())

        q_des = traj.value(local_t).ravel()

        output.SetFromVector(q_des)
