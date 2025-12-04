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

    def __init__(self, plant: MultibodyPlant):
        super().__init__()

        self._plant = plant

    def CalcPosOutput(self, context: Context, output: BasicVector):
        pass
