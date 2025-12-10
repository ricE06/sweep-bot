from manipulation.station import LoadScenario
from pathlib import Path
import random
import re
import numpy as np

def add_cubes(n, x_lower, x_upper, y_lower, y_upper):
    """
    Returns section of directives adding n small
    5 cm cubes within the given bounds
    """
    cube_directives = ""

    for i in range(n):
        x = random.uniform(x_lower, x_upper)
        y = random.uniform(y_lower, y_upper)
        # z = i / 30 # avoid cubes overlapping
        z = 0.025

        cube_directives += f"""
- add_model:
    name: cube_{i}
    file: file://models/cube.sdf
    default_free_body_pose:
        cube_link:
            translation: [{x}, {y}, {z}]
            rotation: !Rpy {{ deg: [0, 0, 0] }}
"""
    return cube_directives

def replace_starting_pose(scenario_string: str, q0: np.ndarray):
    for i in range(7):
        start = str(q0[i])
        scenario_string = re.sub(f"iiwa_joint_{i+1}: \\[[\\d.]+\\]", 
                                 f"iiwa_joint_{i+1}: [{start}]", 
                                 scenario_string)
    return scenario_string


def load_scenario(use_cubes=True, use_position=False, use_weld=False, q0=None):
    """
    Loads the scenario for the sweeper simulation.
    Should contain:
    - the robot
    - on a very large table
    - with a broom at a randomized location, in the third quadrant
    - with many blocks at a randomized location, in the first quadrant
    - a colored region of the table
    - cameras making point clouds for the broom and for the blocks
    Returns the Scenario object.
    """
    if use_weld:
        with open('./scripts/scenario_welded_broom.yaml', 'r') as f:
            scenario_string = f.read()
    else:
        with open('./scripts/scenario.yaml', 'r') as f:
            scenario_string = f.read()

    with open('./scripts/camera.yaml', 'r') as f:
        camera_string = f.read()
    if use_cubes:
        scenario_string += add_cubes(20, -0.5, 0.5, 0.3, 0.7)
    scenario_string += '\n' + camera_string
    if use_position:
        scenario_string += """
model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
        hand_model_name: broom
    wsg: !SchunkWsgDriver {}
    """
    else:
        scenario_string += """
model_drivers:
    wsg: !SchunkWsgDriver {}
"""
    # put in cwd
    scenario_string = scenario_string.replace('file://', f'file://{Path.cwd()}/')
    # print(scenario_string)
    scenario = LoadScenario(data=scenario_string)
    return scenario
