import numpy as np
from pydrake.all import (
        BasicVector,
        Context,
        Diagram,
        MultibodyPlant,
        LeafSystem,
)

from .point_cloud import get_point_cloud

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

    PHASE_OPT      = 0
    PHASE_PREGRIP  = 1
    PHASE_SWEEP    = 2
    PHASE_RETURN   = 3

    def __init__(self, 
                 plant: MultibodyPlant, 
                 station,
                 diagram,
                 traj_opt,
                 traj_pregrip_to_grip,
                 traj_sweep,
                 traj_return=None):

        super().__init__()

        # for the various helper classes to use
        self.diagram = diagram
        self.plant = plant
        self.station = station
        self.context = diagram.CreateDefaultContext()
        self.plant_context = plant.GetMyContextFromRoot(self.context)
        # self.station_context = station.CreateDefaultContext()
        self.base = plant.GetBodyByName("base", plant.GetModelInstanceByName('iiwa'))
        self.body = plant.GetBodyByName("handle_link")

        # link camera
        print(get_point_cloud(self))
        return

        self._nq = plant.num_positions()

        # Store trajectories
        self._traj = {
            self.PHASE_OPT: traj_opt,
            self.PHASE_PREGRIP: traj_pregrip_to_grip,
            self.PHASE_SWEEP: traj_sweep,
            self.PHASE_RETURN: traj_return,
        }

        # state vars:
        # 1. Phase index (discrete integer)
        self._phase_idx = self.DeclareDiscreteState(1)
        # 2. Local time within the phase
        self._phase_time = self.DeclareDiscreteState(1)

        # Input: 1 → request next phase; 0 → do nothing
        self._advance_port = self.DeclareVectorInputPort(
            "advance_phase", BasicVector(1)
        ).get_index()
        # Output: desired joint position
        self.DeclareVectorOutputPort(
            "position_command",
            BasicVector(self._nq),
            self.CalcPosOutput
        )

        # Update handler (discrete time step)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.001,
            handler=self.DoUpdate
        )

    #UPDATE LOOP (state machine)
    def DoUpdate(self, context, state):

        phase = int(context.get_discrete_state(self._phase_idx).get_value()[0])
        t_local = context.get_discrete_state(self._phase_time).get_value()[0]

        advance = 0
        if self.get_input_port(self._advance_port).HasValue(context):
            advance = int(self.get_input_port(self._advance_port).Eval(context)[0])

        traj = self._traj.get(phase, None)

        # Advance loc time each tick
        dt = 0.001
        t_local += dt

        # 1. external request to advance
        if advance == 1:
            phase = min(phase + 1, self.PHASE_RETURN)
            t_local = 0.0

        # 2. Automatic transition on trajectory completion
        elif traj is not None and t_local >= traj.end_time():
            phase = min(phase + 1, self.PHASE_RETURN)
            t_local = 0.0

        # Write back updated states
        state.get_mutable_discrete_state(self._phase_idx).set_value([phase])
        state.get_mutable_discrete_state(self._phase_time).set_value([t_local])

    def CalcPosOutput(self, context, output):
        phase = int(context.get_discrete_state(self._phase_idx).get_value()[0])
        t_local = context.get_discrete_state(self._phase_time).get_value()[0]

        traj = self._traj.get(phase, None)

        if traj is None:
            # If no return trajectory → hold last pose of sweep
            traj = self._traj[self.PHASE_SWEEP]
            t_local = traj.end_time()

        t_local = np.clip(t_local, traj.start_time(), traj.end_time())
        q_des = traj.value(t_local).ravel()

        output.SetFromVector(q_des)

    def generate_sweep():
        pass

# use a state machine to keep track of logic
# diffik to get joint traj from broom traj
# weld broom to robot to test sweeping
