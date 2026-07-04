import unittest

import numpy as np

from src import parallel_env
from src.geometry import Rect, distance, line_of_sight_clear, segment_intersects_rect
from src.instance_loader import load_instance
from src.map_generator import generate_valid_map, validate_map


class HoldTheLineTests(unittest.TestCase):
    def test_reset_observation_shapes(self):
        env = parallel_env()
        observations, infos = env.reset(seed=123)

        self.assertEqual(env.agents, ["blue_0", "blue_1", "red_0", "red_1"])
        self.assertEqual(observations["blue_0"].shape, (26,))
        self.assertEqual(observations["red_0"].shape, (21,))
        self.assertTrue(infos["blue_0"]["alive"])

    def test_step_accepts_continuous_actions(self):
        env = parallel_env(max_episode_steps=5)
        env.reset(seed=123)
        actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)

        self.assertEqual(set(rewards), set(env.possible_agents))
        self.assertEqual(observations["blue_0"].shape, (26,))
        self.assertFalse(any(terminations.values()))
        self.assertFalse(any(truncations.values()))
        self.assertIn("detected_reds", infos["blue_0"])

    def test_distance_computation(self):
        self.assertEqual(
            distance(
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([3.0, 4.0], dtype=np.float32),
            ),
            5.0,
        )

    def test_segment_intersects_rectangle(self):
        blocker = Rect(4.0, 4.0, 2.0, 2.0)

        self.assertTrue(
            segment_intersects_rect(
                np.array([0.0, 5.0], dtype=np.float32),
                np.array([10.0, 5.0], dtype=np.float32),
                blocker,
            )
        )
        self.assertFalse(
            segment_intersects_rect(
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([10.0, 1.0], dtype=np.float32),
                blocker,
            )
        )

    def test_line_of_sight_blocked_by_rectangle(self):
        blocker = Rect(4.0, 4.0, 2.0, 2.0)

        self.assertFalse(
            line_of_sight_clear(
                np.array([0.0, 5.0], dtype=np.float32),
                np.array([10.0, 5.0], dtype=np.float32),
                [blocker],
            )
        )
        self.assertTrue(
            line_of_sight_clear(
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([10.0, 1.0], dtype=np.float32),
                [blocker],
            )
        )

    def test_detection_radius(self):
        env = parallel_env()
        env.reset(seed=123)
        env._states["blue_0"].pos = np.array([50.0, 10.0], dtype=np.float32)
        env._states["red_0"].pos = np.array([50.0, 25.0], dtype=np.float32)

        self.assertTrue(env.blue_can_detect_red("blue_0", "red_0"))

        env._states["red_0"].pos = np.array([50.0, 29.5], dtype=np.float32)
        self.assertFalse(env.blue_can_detect_red("blue_0", "red_0"))

    def test_destroyed_after_sustained_tether(self):
        # Destruction isn't instant-on-contact anymore: a red drone has to stay
        # continuously tethered (in range + line of sight) for destroy_time
        # seconds before it's destroyed.
        env = parallel_env()
        env.reset(seed=123)
        env._states["blue_0"].pos = np.array([50.0, 20.0], dtype=np.float32)
        env._states["red_0"].pos = np.array([51.0, 20.0], dtype=np.float32)
        env._states["red_1"].pos = np.array([50.0, 95.0], dtype=np.float32)

        actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
        steps = int(env.blue_destroy_time / env.dt) + 1
        rewards = {}
        for _ in range(steps):
            if not env.agents:
                break
            observations, rewards, terminations, truncations, infos = env.step(actions)

        self.assertFalse(env._states["red_0"].alive)
        self.assertTrue(env._states["red_1"].alive)
        self.assertLess(rewards["red_0"], 0.0)

    def test_tether_break_resets_exposure(self):
        # Breaking the tether before destroy_time elapses (e.g. by moving out
        # of detection range, standing in for ducking behind a building) fully
        # resets the exposure clock rather than just pausing it.
        env = parallel_env()
        env.reset(seed=123)
        env._states["blue_0"].pos = np.array([50.0, 20.0], dtype=np.float32)
        env._states["red_0"].pos = np.array([51.0, 20.0], dtype=np.float32)
        env._states["red_1"].pos = np.array([50.0, 95.0], dtype=np.float32)
        actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}

        env.step(actions)
        self.assertGreater(env._states["red_0"].exposure, 0.0)

        env._states["red_0"].pos = np.array([50.0, 95.0], dtype=np.float32)
        env.step(actions)
        self.assertEqual(env._states["red_0"].exposure, 0.0)
        self.assertTrue(env._states["red_0"].alive)

    def test_blue_wins_when_all_reds_intercepted(self):
        env = parallel_env()
        env.reset(seed=123)

        for red_agent in env.red_agents:
            env._states[red_agent].pos = env._states["blue_0"].pos.copy()

        actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
        steps = int(env.blue_destroy_time / env.dt) + 1
        for _ in range(steps):
            observations, rewards, terminations, truncations, infos = env.step(actions)
            if not env.agents:
                break

        self.assertEqual(observations, {})
        self.assertTrue(all(terminations.values()))
        self.assertFalse(any(truncations.values()))
        self.assertGreater(rewards["blue_0"], 0.0)
        self.assertLess(rewards["red_0"], 0.0)

    def test_red_succeeds_after_scouting_threshold(self):
        env = parallel_env()
        env.reset(seed=123)
        red = "red_0"
        env._states[red].pos = env.assets[0].center_array.copy()
        env._states[red].info_collected = env.red_info_threshold - env.dt

        actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        self.assertEqual(observations, {})
        self.assertTrue(all(terminations.values()))
        self.assertGreater(rewards[red], 0.0)
        self.assertLess(rewards["blue_0"], 0.0)

    def test_scouting_accumulates_before_threshold(self):
        env = parallel_env()
        env.reset(seed=123)
        red = "red_0"
        env._states[red].pos = env.assets[0].center_array.copy()

        actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        self.assertAlmostEqual(env._states[red].info_collected, env.dt)
        self.assertFalse(any(terminations.values()))
        self.assertGreater(rewards[red], 0.0)

    def test_collision_with_building_reverts_position(self):
        env = parallel_env()
        env.reset(seed=123)
        agent = "blue_0"
        env._states[agent].pos = np.array([17.0, 50.8], dtype=np.float32)
        env._states[agent].vel = np.array([0.0, 18.0], dtype=np.float32)
        old_pos = env._states[agent].pos.copy()

        actions = {agent_id: np.zeros(2, dtype=np.float32) for agent_id in env.agents}
        env.step(actions)

        np.testing.assert_allclose(env._states[agent].pos, old_pos)
        np.testing.assert_allclose(env._states[agent].vel, np.zeros(2, dtype=np.float32))

    def test_reset_spawn_positions_inside_correct_zones(self):
        env = parallel_env()
        env.reset(seed=123)

        for blue_agent in env.blue_agents:
            self.assertTrue(env.map_config.blue_spawn_zone.contains_point(env._states[blue_agent].pos))
        for red_agent in env.red_agents:
            self.assertTrue(
                any(zone.contains_point(env._states[red_agent].pos) for zone in env.map_config.red_spawn_zones)
            )
            self.assertGreater(env._states[red_agent].pos[1], env.world_size[1] * 0.85)

    def test_spawn_positions_are_evenly_spaced_across_bands(self):
        env = parallel_env(map_config=load_instance("test2"))
        env.reset(seed=123)

        blue_xs = [env._states[agent].pos[0] for agent in env.blue_agents]
        red_xs = [env._states[agent].pos[0] for agent in env.red_agents]

        np.testing.assert_allclose(blue_xs, [6.0, 114.0], atol=1e-5)
        np.testing.assert_allclose(np.diff(red_xs), np.full(2, 54.05), atol=1e-4)

    def test_render_state_includes_team_shared_visibility(self):
        env = parallel_env()
        env.reset(seed=123)
        env._states["blue_0"].pos = np.array([50.0, 10.0], dtype=np.float32)
        env._states["red_0"].pos = np.array([50.0, 24.0], dtype=np.float32)
        env._last_detections = env._compute_detections()

        visibility = env.render_state()["team_visibility"]

        self.assertIn("red_0", visibility["blue_visible_reds"])

    def test_generated_map_validates_and_runs(self):
        config = generate_valid_map(seed=5, building_count=7)
        result = validate_map(config)
        self.assertTrue(result.valid, result.reasons)
        self.assertGreaterEqual(result.approach_routes, 2)

        env = parallel_env(map_config=config, max_episode_steps=3)
        observations, infos = env.reset(seed=123)
        self.assertEqual(set(observations), set(env.possible_agents))

        actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        self.assertEqual(set(rewards), set(env.possible_agents))

    def test_rgb_array_render_returns_image(self):
        env = parallel_env(render_mode="rgb_array")
        env.reset(seed=123)

        image = env.render()

        self.assertEqual(image.shape, (700, 700, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(image.max(), image.min())
        env.close()


if __name__ == "__main__":
    unittest.main()
