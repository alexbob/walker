import unittest

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from humanoid_env import make_env


class HumanoidTests(unittest.TestCase):
    def test_environment_and_viewer_info(self):
        env = make_env()
        try:
            check_env(env, warn=False)
            self.assertEqual(env.action_space.shape, (17,))
            head = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, 'head')
            self.assertGreaterEqual(head, 0)
            obs, info = env.reset(seed=42)
            self.assertIn('x_position', info)
            for _ in range(100):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(np.isfinite(reward))
                self.assertIn('x_position', info)
                if terminated or truncated:
                    obs, info = env.reset()
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
