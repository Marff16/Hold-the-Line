import unittest

from src.core.instance_loader import list_instances, load_instance
from src.server.app import ControlRequest, WebSimulation


class WebAppTests(unittest.TestCase):
    def test_web_simulation_snapshot_and_controls(self):
        simulation = WebSimulation()
        simulation.load_instance("instance1")

        snapshot = simulation.snapshot()
        self.assertEqual(snapshot["step"], 0)
        self.assertEqual(snapshot["controls"]["speed"], 1)
        # "Learned Blue" only appears once a trained checkpoint exists on
        # disk (checkpoints/blue_actor_critic.pt) - don't hard-fail the test
        # depending on whether training has run in this environment.
        self.assertIn("Random", snapshot["controls"]["policy_options_blue"])
        self.assertIn("Avoider", snapshot["controls"]["policy_options_blue"])
        self.assertIn("Random", snapshot["controls"]["policy_options_red"])
        self.assertIn("Avoider", snapshot["controls"]["policy_options_red"])
        self.assertNotIn("Learned Blue", snapshot["controls"]["policy_options_red"])
        self.assertNotIn("Learned Red", snapshot["controls"]["policy_options_blue"])
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
        self.assertGreaterEqual(len(instances), 1)

        instance1 = load_instance("instance1")
        self.assertEqual(instance1.name, "Instance 1")
        self.assertEqual(instance1.blue_drones.count, 2)
        self.assertEqual(instance1.red_drones.count, 2)
        self.assertEqual(len(instance1.assets), 2)

        simulation = WebSimulation()
        simulation.load_instance("instance1")
        snapshot = simulation.snapshot()
        self.assertEqual(snapshot["controls"]["instance_id"], "instance1")
        self.assertEqual(len(snapshot["agents"]), 4)

    def test_terrain_toggle_control(self):
        simulation = WebSimulation()
        simulation.apply_control(ControlRequest(terrain_enabled=True))

        snapshot = simulation.snapshot()

        self.assertTrue(snapshot["terrain"]["enabled"])
        self.assertIn("available", snapshot["terrain"])


if __name__ == "__main__":
    unittest.main()
