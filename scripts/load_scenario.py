from manipulation.station import LoadScenario
from pathlib import Path
import random

def add_cubes(n, x_lower, x_upper, y_lower, y_upper):
    """
    Returns section of directives adding n small
    5 cm cubes within the given bounds
    """
    cube_directives = ""

    for i in range(n):
        x = random.uniform(x_lower, x_upper)
        y = random.uniform(y_lower, y_upper)
        z = i / 30 # avoid cubes overlapping

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

def load_scenario():
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
    with open('./scripts/scenario.yaml', 'r') as f:
        scenario_string = f.read()
    with open('./scripts/camera.yaml', 'r') as f:
        camera_string = f.read()
    scenario_string += add_cubes(20, -1, +1, 0.5, 1)
    scenario_string += '\n' + camera_string
    # put in cwd
    scenario_string = scenario_string.replace('file://', f'file://{Path.cwd()}/')
    # print(scenario_string)
    scenario = LoadScenario(data=scenario_string)
    return scenario
