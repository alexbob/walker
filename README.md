# Two-legged walker

Use the project virtual environment:

```sh
.venv/bin/python train.py --timesteps 1000000
.venv/bin/python watch.py
.venv/bin/python -m unittest test_walking_env.py
```

Training resumes `walker2d_both_legs.zip` if present, otherwise starts a fresh PPO policy. Older checkpoints are preserved. The new policy has five extra observations for reward memory and cannot load the older checkpoint directly.

`walking_env.py` keeps Walker2d's forward movement, health, and control effort rewards, and adds:

- A penalty of 1 per unused foot per simulation step after that foot has not touched the floor for 1 second.
- A bonus up to 0.5 for changing the supporting foot, after 0.15 seconds of continuous single-foot support. The first support earns no bonus. The bonus is multiplied by forward velocity clipped to [0, 1], so standing or moving backward earns no switch bonus.

Tune `unused_after`, `unused_penalty`, `switch_bonus`, and `min_support_time` in `BothLegsReward` to change these incentives. They encourage both feet to participate but do not enforce a natural gait; contact tapping or shuffling may still emerge. Evaluate behavior in the viewer after training.

The viewer uses the same reward wrapper and observation shape as training. `both_legs.monitor.csv` records episode returns and the final step's reward components (not component totals). Model files, the virtual environment, and generated logs are ignored by Git.
