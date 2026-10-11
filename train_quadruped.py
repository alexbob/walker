from quadruped_env import MODEL_PATH, make_env
from training import run_training


if __name__ == "__main__":
    run_training(make_env, MODEL_PATH, "four-legged Ant")
