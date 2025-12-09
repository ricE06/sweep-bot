from dataclasses import dataclass
from typing import Optional
import numpy as np
from pydrake.all import (
        AbstractValue,
        BasicVector,
        Context,
        DerivativeTrajectory,
        DiscreteUpdateEvent,
        DiscreteValues,
        Diagram,
        MultibodyPlant,
        LeafSystem,
        Trajectory,
        WitnessFunction,
        WitnessFunctionDirection
)

from .point_cloud import get_point_cloud
from .trajectory_helpers import TrajectoryGenerator, MoveToPregrip, PregripToGrip, ManipulateBroom
from .ik import solve_ik_for_pose

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

@dataclass
class Trajectories:
    gripper_pos: Trajectory
    gripper_vel: Trajectory
    wsg: Trajectory

class MetaController(LeafSystem):
    """
    Controller that provides commands to the iiwa
    depending on where in the simulation we are in.
    """

    START = 0
    PREGRIP = 1
    GRIP = 2
    PHASE_RETURN = 3

    diagram: Diagram = None
    context: Context = None
    plant_context: Context = None

    def __init__(self, plant: MultibodyPlant, station):

        super().__init__()

        # for the various helper classes to use
        self.plant = plant
        self.station = station
        # self.station_context = station.CreateDefaultContext()
        self.base = plant.GetBodyByName("base", plant.GetModelInstanceByName('iiwa'))
        self.body = plant.GetBodyByName("handle_link")
        self.gripper = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))

        self._nq = 7

        self._discrete_state = self.DeclareDiscreteState(3)
        self._phase_idx = self._discrete_state
        self._last_traj_start_time = int(self._discrete_state) + 1
        self._last_traj_end_time = int(self._discrete_state) + 2

        # trajectory and first derivative and wsg
        # type Trajectories | None
        self._cur_sweep_trajectory = self.DeclareAbstractState(AbstractValue.Make(None))

        # Concatenate joint positions, velocities and wsg to be fed into Demultiplexer
        self.DeclareVectorOutputPort(
            "position_and_wsg",
            BasicVector(2*self._nq + 1),
            self.CalcSysOutput
        )

        # self.DeclareInitializationDiscreteUpdateEvent(self.UpdateTrajectory)
        self.DeclarePerStepDiscreteUpdateEvent(self.UpdateTrajectory)

    def add_diagram(self, diagram: Diagram):
        self.diagram = diagram
        self.context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        # link camera
        print(get_point_cloud(self))
        ManipulateBroom().trajectory(self)

    def UpdateTrajectory(self, context: Context, values: DiscreteValues):
        time = context.get_time()
        state = context.get_mutable_discrete_state(int(self._discrete_state))
        phase, start_time, end_time = state.get_value()
        if end_time > time:
            return

        values_vec = values.get_mutable_vector()

        phase_vec = context.get_mutable_discrete_state(int(self._phase_idx))
        phase = phase_vec.get_value()[0]

        if phase == self.START:
            print(f'called at time {time}')
            print('start!')
            traj_gen = MoveToPregrip()
            values_vec.SetAtIndex(int(self._phase_idx), self.PREGRIP) # will be at pregrip once done
        elif phase == self.PREGRIP:
            traj_gen = PregripToGrip()
            print(f'called at time {time}')
            print('moved to pregrip!')
        elif phase == self.GRIP:
            pass

        trajectory, wsg_trajectory = traj_gen.trajectory(self)
        traj_deriv = trajectory.MakeDerivative(1)
        trajectories = Trajectories(trajectory, traj_deriv, wsg_trajectory)
        traj_state = context.get_mutable_abstract_state(int(self._cur_sweep_trajectory))
        traj_state.set_value(trajectories)

        print(values_vec)
        traj_length = trajectory.end_time()
        print(f'will end at: {time+traj_length}')
        values_vec.SetAtIndex(int(self._last_traj_start_time), time)
        values_vec.SetAtIndex(int(self._last_traj_end_time), time+traj_length)

        # start_time_port = context.get_mutable_discrete_state(int(self._last_traj_start_time))
        # start_time_port.set_value(np.array([time]))
        # end_time_port.set_value(np.array([time+traj_length]))

        # START -> trajectory opt to pregrip
        # found pregrip -> trajectory to grip
        # grip -> trajectory to manipulate broom
        # finished manipulating -> trajectory to pregrip

    def CalcSysOutput(self, context: Context, output: BasicVector):
        time = context.get_time()
        # start_time_port = context.get_mutable_discrete_state(int(self._last_traj_start_time))
        # start_time = start_time_port.get_value()[0]
        state = context.get_mutable_discrete_state(int(self._discrete_state))
        phase, start_time, end_time = state.get_value()
        rel_time = time - start_time

        trajectories: Trajectories = context.get_abstract_state(int(self._cur_sweep_trajectory)).get_value()
        if trajectories is None:
            # has not initialized yet
            print(f'empty trajectories at time {time}')
            output.SetFromVector(np.zeros(15,))
            return

        pos = trajectories.gripper_pos
        vel = trajectories.gripper_vel
        t_local = np.clip(rel_time, pos.start_time(), pos.end_time())

        # don't ask
        q_des = np.array(pos.value(t_local).ravel()).reshape(1, -1).reshape(7,)
        q_dot_des = np.array(vel.value(t_local).ravel()).reshape(1, -1).reshape(7,)
        wsg_val = trajectories.wsg.value(t_local).ravel()[0]
        wsg_des = np.array([float(wsg_val)])

        output.SetFromVector(np.concat([q_des, q_dot_des, wsg_des]))

    #UPDATE LOOP (state machine)
    def DoUpdate(self, context, state):

        phase = int(context.get_discrete_state(self._phase_idx).get_value()[0])

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


# use a state machine to keep track of logic
# diffik to get joint traj from broom traj
# weld broom to robot to test sweeping

