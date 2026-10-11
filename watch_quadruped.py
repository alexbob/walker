from quadruped_env import MODEL_PATH, make_env
from watch import run_viewer

if __name__ == "__main__":
    run_viewer(make_env, MODEL_PATH, "four-legged Ant", "train_quadruped.py")
