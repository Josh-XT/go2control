import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "go2control"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api_server import RobotConnection  # noqa: E402
from config import AppConfig  # noqa: E402


class FakeSportClient:
    def __init__(self, state, code=0):
        self.state = state
        self.code = code

    def GetState(self):
        return self.code, self.state


class ApiServerStatusTests(unittest.TestCase):
    def test_simulation_battery_uses_simulated_state(self):
        robot = RobotConnection(AppConfig(simulation=True))

        self.assertEqual(robot.get_battery_percent(), 85)

    def test_dds_battery_reads_sdk_state_and_clamps(self):
        robot = RobotConnection(AppConfig(simulation=False))
        robot._sport_client = FakeSportClient({"battery_level": 106.4})

        self.assertEqual(robot.get_battery_percent(), 100)

    def test_battery_falls_back_to_last_known_value_after_read_failure(self):
        robot = RobotConnection(AppConfig(simulation=False))
        robot._sport_client = FakeSportClient({"battery_level": 72.2})
        self.assertEqual(robot.get_battery_percent(), 72)

        robot._sport_client = FakeSportClient({}, code=1)
        self.assertEqual(robot.get_battery_percent(), 72)


if __name__ == "__main__":
    unittest.main()
