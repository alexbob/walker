"""Shared PPO runner with Windows-safe parallel simulation and CUDA selection."""
import argparse
from functools import partial
import multiprocessing
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install CUDA-enabled PyTorch using the official "
                         "PyTorch Windows selector, or use --device cpu.")
    return "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"


def worker_env(make_env, log_dir, rank, seed, torch_threads):
    torch.set_num_threads(torch_threads)
    env = Monitor(make_env(), filename=str(log_dir / f"worker_{rank}"))
    env.reset(seed=seed + rank)
    env.action_space.seed(seed + rank)
    return env


def run_training(make_env, default_model_path, label):
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=f"Train {label} with PPO")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--n-envs", type=int, default=1, help="Parallel CPU simulations")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout steps per environment")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100_000, help="Checkpoint interval in total environment steps")
    parser.add_argument("--model-path", type=Path, default=default_model_path)
    args = parser.parse_args()
    for name in ("timesteps", "n_envs", "n_steps", "torch_threads", "save_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        device = select_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    batch_size = args.batch_size if args.batch_size is not None else (256 if device == "cuda" else 64)
    rollout_size = args.n_steps * args.n_envs
    if batch_size <= 1 or rollout_size <= 1 or rollout_size % batch_size:
        parser.error("batch size must exceed 1 and evenly divide n-steps * n-envs")
    model_path = args.model_path.expanduser().resolve()
    if model_path.suffix != ".zip":
        parser.error("--model-path must end in .zip")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = model_path.parent / "logs" / model_path.stem
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    print(f"Device: {device}; CPU simulation workers: {args.n_envs}; "
          f"rollout: {rollout_size}; batch: {batch_size}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}; PyTorch CUDA: {torch.version.cuda}")
    factories = [partial(worker_env, make_env, log_dir, rank, args.seed, args.torch_threads)
                 for rank in range(args.n_envs)]
    env = (DummyVecEnv(factories) if args.n_envs == 1
           else SubprocVecEnv(factories, start_method="spawn"))
    try:
        options = dict(n_steps=args.n_steps, batch_size=batch_size, seed=args.seed)
        if model_path.exists():
            print(f"Continuing {label} from {model_path}")
            model = PPO.load(model_path, env=env, device=device, **options)
        else:
            print(f"Training new {label}")
            model = PPO("MlpPolicy", env, device=device, verbose=1, **options)
        callback = CheckpointCallback(save_freq=max(args.save_every // args.n_envs, 1),
                                      save_path=str(log_dir / "checkpoints"),
                                      name_prefix=model_path.stem)
        try:
            model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False, callback=callback)
        except KeyboardInterrupt:
            print("Training interrupted; saving current policy.")
        model.save(model_path)
        print(f"Saved model to {model_path}")
    finally:
        env.close()
