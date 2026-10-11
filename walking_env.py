"""Shared walker environment with configurable incentives to use both feet."""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "walker2d_both_legs_30kg_ankle10.zip"


class BothLegsReward(gym.Wrapper):
    """Penalize unused feet and reward forward-moving support changes.

    Contact ages and support state are observations so PPO can see the
    reward's memory. Original Walker2d health and effort rewards remain.
    """

    def __init__(self, env, unused_after=1.0, unused_penalty=1.0,
                 switch_bonus=0.5, min_support_time=0.15):
        super().__init__(env)
        if unused_after <= 0 or min_support_time <= 0:
            raise ValueError("Timing thresholds must be positive")
        if unused_penalty < 0 or switch_bonus < 0:
            raise ValueError("Reward weights must be nonnegative")
        self.unused_after = unused_after
        self.unused_penalty = unused_penalty
        self.switch_bonus = switch_bonus
        self.min_support_time = min_support_time
        model = env.unwrapped.model
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                     for name in ("foot_geom", "foot_left_geom")]
        if min([self.floor, *self.feet]) < 0:
            raise ValueError("Expected named floor and foot geometries")
        low = np.concatenate([env.observation_space.low, np.zeros(5)])
        high = np.concatenate([env.observation_space.high, np.ones(5)])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)
        self._clear()

    def _clear(self):
        self.ages = np.zeros(2)
        self.last_support = -1
        self.candidate = -1
        self.support_time = 0.0

    def _observe(self, obs):
        memory = [*np.minimum(self.ages / self.unused_after, 1),
                  (self.last_support + 1) / 2,
                  (self.candidate + 1) / 2,
                  min(self.support_time / self.min_support_time, 1)]
        return np.concatenate([obs, memory])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._clear()
        return self._observe(obs), info

    def _contacts(self):
        contacts = [False, False]
        for contact in self.env.unwrapped.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            for index, foot in enumerate(self.feet):
                if pair == {self.floor, foot} and contact.dist <= 0:
                    contacts[index] = True
        return contacts

    def _update_reward(self, contacts, velocity):
        dt = self.env.unwrapped.dt
        self.ages = np.where(contacts, 0.0, self.ages + dt)
        penalty = self.unused_penalty * float(np.sum(self.ages > self.unused_after))
        support = contacts.index(True) if sum(contacts) == 1 else -1
        bonus = 0.0
        if support < 0:
            self.candidate = -1
            self.support_time = 0.0
        else:
            if support != self.candidate:
                self.candidate = support
                self.support_time = 0.0
            self.support_time += dt
            if self.support_time >= self.min_support_time and support != self.last_support:
                if self.last_support >= 0:
                    bonus = self.switch_bonus * float(np.clip(velocity, 0, 1))
                self.last_support = support
        return bonus, penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        contacts = self._contacts()
        bonus, penalty = self._update_reward(contacts, info["x_velocity"])
        info.update(reward_support_switch=bonus, reward_unused_leg=-penalty,
                    right_foot_contact=int(contacts[0]), left_foot_contact=int(contacts[1]))
        return self._observe(obs), reward + bonus - penalty, terminated, truncated, info


def make_env(**kwargs):
    return BothLegsReward(gym.make("Walker2d-v5", xml_file=str(ROOT / "walker2d_separated.xml"), **kwargs))
