"""Independent Humanoid-v5 baseline with arms, legs, and a fixed head."""
from pathlib import Path

import gymnasium as gym

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "humanoid_ppo.zip"


def make_env(**kwargs):
    return gym.make("Humanoid-v5", xml_file=str(ROOT / "humanoid.xml"), **kwargs)
