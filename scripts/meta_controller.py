from dataclasses import dataclass
from typing import Optional
import numpy as np
from pydrake.all import (
        AbstractValue,
        BasicVector,
        Context,
        DerivativeTrajectory,
        DiagramBuilder,
        DiscreteUpdateEvent,
        DiscreteValues,
        Diagram,
        Integrator,
        MultibodyPlant,
        LeafSystem,
        RigidTransform,
        Simulator,
        Trajectory,
        TrajectorySource,
)
from manipulation.station import MakeHardwareStation

from .point_cloud import get_point_cloud
from .trajectory_helpers import TrajectoryGenerator, MoveToPregrip, PregripToGrip, ManipulateBroom
from .diff_ik import PseudoInverseController
from .load_scenario import load_scenario

@dataclass
class Trajectories:
    gripper_pos: Trajectory
    gripper_vel: Trajectory
    wsg: Trajectory

def make_pinv_system(traj_v_src: TrajectorySource, q0: np.ndarray):
    builder = DiagramBuilder() 
    scenario = load_scenario(use_cubes=False)

    station = MakeHardwareStation(scenario)  
    plant: MultibodyPlant = station.plant()

    context = station.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    iiwa = plant.GetModelInstanceByName('iiwa')
    plant.SetPositions(plant_context, iiwa, q0)

    pinv_control: PseudoInverseController = builder.AddSystem(PseudoInverseController(plant))
    integrator = builder.AddSystem(Integrator(7))

    builder.Connect(traj_v_src.get_output_port(), pinv_control.V_G_port)
    builder.Connect(pinv_control.get_output_port(),
                    integrator.get_input_port())

    diagram = builder.Build()
    sim = Simulator(diagram)
    return sim

class MetaController(LeafSystem):
    """
    Controller that provides commands to the iiwa
    depending on where in the simulation we are in.
    """

    START = 0
    PREGRIP = 1
    GRIP = 2
    PHASE_RETURN = 3

    TRAJ_Q = 0
    TRAJ_POSE = 1

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

        self._discrete_state = self.DeclareDiscreteState(4)
        self._phase_idx = self._discrete_state
        self._last_traj_start_time = int(self._discrete_state) + 1
        self._last_traj_end_time = int(self._discrete_state) + 2
        self._traj_mode = int(self._discrete_state) + 3

        # trajectory and first derivative and wsg
        # type Trajectories | None
        self._cur_sweep_trajectory = self.DeclareAbstractState(AbstractValue.Make(None))
        self._cur_pinv_simulator = self.DeclareAbstractState(AbstractValue.Make(None))

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

        phase_vec = context.get_mutable_discrete_state(int(self._phase_idx))
        phase = phase_vec.get_value()[0]

        use_pinv = False
        if phase == self.START:
            print(f'called at time {time}')
            print('start!')
            traj_gen = MoveToPregrip()
            values_vec.SetAtIndex(int(self._phase_idx), self.PREGRIP) # will be at pregrip once done
        elif phase == self.PREGRIP:
            traj_gen = PregripToGrip()
            print(f'called at time {time}')
            print('moved to pregrip!')
            use_pinv = True
            # from gripper vel -> joint ang
        elif phase == self.GRIP:
            pass

        trajectory, wsg_trajectory = traj_gen.trajectory(self)
        traj_deriv = trajectory.MakeDerivative(1)
        trajectories = Trajectories(trajectory, traj_deriv, wsg_trajectory)
        traj_state = context.get_mutable_abstract_state(int(self._cur_sweep_trajectory))
        traj_state.set_value(trajectories)

        if use_pinv:
            iiwa = self.plant.GetModelInstanceByName('iiwa')
            q0 = self.plant.GetPositions(self.plant_context, iiwa)
            pinv_sim = make_pinv_system(TrajectorySource(traj_deriv), q0)
            sim_state = context.get_mutable_abstract_state(int(self._cur_pinv_simulator))
            sim_state.set_value(pinv_sim)

        print(values_vec)
        traj_length = trajectory.end_time()
        print(f'will end at: {time+traj_length}')
        values_vec.SetAtIndex(int(self._last_traj_start_time), time)
        values_vec.SetAtIndex(int(self._last_traj_end_time), time+traj_length)
        pinv_state_value = self.TRAJ_POSE if use_pinv else self.TRAJ_Q
        values_vec.SetAtIndex(int(self._traj_mode), pinv_state_value)

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

