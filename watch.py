from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = Path(__file__).resolve().parent / "walker2d_separated.zip"


def main():
    if MODEL_PATH.exists():
        model = PPO.load(MODEL_PATH, device="cpu")
    else:
        model = None
        print("No trained Walker2d model yet. Showing an untrained two-leg preview.")
        print("Run python train.py to train it to walk.")

    env = gym.make("Walker2d-v5", xml_file=str(Path(__file__).resolve().parent / "walker2d_separated.xml"), render_mode="human", camera_id=-1, default_camera_config={"azimuth": 135, "elevation": -15, "distance": 4.0})

    try:
        obs, info = env.reset()
        episode_reward = 0.0

        while True:
            if model is None:
                action = env.action_space.low * 0.0
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)

            if terminated or truncated:
                print(f"Episode reward: {episode_reward:.1f}")
                episode_reward = 0.0
                obs, info = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
