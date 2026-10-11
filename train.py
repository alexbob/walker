import argparse

from walking_env import MODEL_PATH, make_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    env = Monitor(make_env(), filename="both_legs.monitor.csv",
                  info_keywords=("reward_support_switch", "reward_unused_leg"))

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
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
        )
        model.save(MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
