from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

MODEL_PATH = Path(__file__).resolve().parent / "walker2d_separated.zip"


def main():
    env = Monitor(gym.make("Walker2d-v5", xml_file=str(Path(__file__).resolve().parent / "walker2d_separated.xml")))

    try:
        if MODEL_PATH.exists():
            print("Continuing training from saved Walker2d model.")
            model = PPO.load(MODEL_PATH, env=env, device="cpu")
        else:
            print("Training a new Walker2d model.")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device="cpu",
            )

        model.learn(
            total_timesteps=1_000_000,
            reset_num_timesteps=False,
        )
        model.save(MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
