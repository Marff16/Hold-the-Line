import unittest

from src.instance_loader import list_instances, load_instance
from src.web_app import ControlRequest, WebSimulation


class WebAppTests(unittest.TestCase):
    def test_web_simulation_snapshot_and_controls(self):
        simulation = WebSimulation()

        snapshot = simulation.snapshot()
        self.assertEqual(snapshot["step"], 0)
        self.assertEqual(snapshot["controls"]["speed"], 1)
        self.assertEqual(snapshot["controls"]["policy_options"], ["Random"])
        self.assertEqual(snapshot["controls"]["policy_blue"], "Random")
        self.assertEqual(snapshot["controls"]["policy_red"], "Random")
        self.assertEqual(len(snapshot["agents"]), 4)

        simulation.apply_control(ControlRequest(playing=True, speed=9, selected_agent="blue_0"))
        snapshot = simulation.snapshot()
        self.assertTrue(snapshot["controls"]["playing"])
        self.assertEqual(snapshot["controls"]["speed"], 9)
        self.assertEqual(snapshot["controls"]["selected_agent"], "blue_0")

        simulation.advance(2)
        self.assertEqual(simulation.snapshot()["step"], 2)

        simulation.reset()
        snapshot = simulation.snapshot()
        self.assertEqual(snapshot["step"], 0)
        self.assertFalse(snapshot["controls"]["playing"])

    def test_instance_loader_and_switching(self):
        instances = list_instances()
        self.assertGreaterEqual(len(instances), 2)

        test2 = load_instance("test2")
        self.assertEqual(test2.name, "Test 2")
        self.assertEqual(test2.blue_drones.count, 2)
        self.assertEqual(test2.red_drones.count, 3)
        self.assertEqual(len(test2.assets), 3)

        simulation = WebSimulation()
        simulation.load_instance("test2")
        snapshot = simulation.snapshot()
        self.assertEqual(snapshot["controls"]["instance_id"], "test2")
        self.assertEqual(len(snapshot["agents"]), 5)

    def test_terrain_toggle_control(self):
        simulation = WebSimulation()
        simulation.apply_control(ControlRequest(terrain_enabled=True))

        snapshot = simulation.snapshot()

        self.assertTrue(snapshot["terrain"]["enabled"])
        self.assertIn("available", snapshot["terrain"])


if __name__ == "__main__":
    unittest.main()
