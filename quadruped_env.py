from pathlib import Path
import gymnasium as gym

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "quadruped_ant.zip"

def make_env(**kwargs):
    return gym.make("Ant-v5", xml_file=str(ROOT / "quadruped.xml"), **kwargs)
