import argparse

from quadruped_env import MODEL_PATH, make_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    env = Monitor(make_env(), filename="quadruped.monitor.csv")

    try:
        if MODEL_PATH.exists():
            print("Continuing training from saved four-legged Ant model.")
            model = PPO.load(MODEL_PATH, env=env, device="cpu")
        else:
            print("Training a new four-legged Ant model.")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device="cpu",
            )

        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
        )
        model.save(MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
