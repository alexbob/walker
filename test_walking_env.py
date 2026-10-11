import unittest
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from walking_env import make_env


class BothLegsTests(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.env.reset(seed=42)

    def tearDown(self):
        self.env.close()

    def hold(self, contacts, velocity=1.0, seconds=0.2):
        results = [self.env._update_reward(contacts, velocity)
                   for _ in range(int(seconds / self.env.unwrapped.dt) + 1)]
        return sum(b for b, _ in results), results[-1][1]

    def test_unused_leg_penalty(self):
        _, penalty = self.hold([True, False], seconds=1.2)
        self.assertEqual(penalty, 1.0)
        _, penalty = self.hold([True, True])
        self.assertEqual(penalty, 0)

    def test_switch_requires_sustained_alternation(self):
        self.assertEqual(self.hold([True, False])[0], 0)
        self.assertEqual(self.hold([True, False])[0], 0)
        self.assertAlmostEqual(self.hold([False, True])[0], 0.5)
        self.assertEqual(self.hold([True, False], velocity=0)[0], 0)
        self.assertEqual(self.hold([True, True])[0], 0)

    def test_reset_and_ppo(self):
        check_env(self.env)
        model = PPO('MlpPolicy', self.env, n_steps=32, batch_size=32,
                    n_epochs=1, device='cpu', verbose=0)
        model.learn(total_timesteps=64)
        obs, _ = self.env.reset(seed=42)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertTrue(np.all(self.env.ages == 0))
        for _ in range(100):
            obs, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(self.env.observation_space.contains(obs))
            if terminated or truncated:
                self.env.reset()


if __name__ == '__main__':
    unittest.main()
