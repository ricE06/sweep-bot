from pydrake.all import (
        LeafSystem,
)


class MetaController(LeafSystem):
    """
    Controller that provides commands to the iiwa 
    depending on where in the simulation we are in.
    """

    def __init__(self):
        pass
