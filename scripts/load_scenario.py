from manipulation.station import LoadScenario
from pathlib import Path

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
    # put in cwd
    scenario_string = scenario_string.replace('file://', f'file://{Path.cwd()}/')
    scenario = LoadScenario(data=scenario_string)
    return scenario