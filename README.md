# Two-legged walker

Use the project virtual environment:

```sh
.venv/bin/python train.py --timesteps 1000000
.venv/bin/python watch.py
.venv/bin/python -m unittest test_walking_env.py
```

Training resumes `walker2d_both_legs_30kg_ankle10.zip` if present, otherwise starts a fresh PPO policy. Older checkpoints are preserved. The new policy has five extra observations for reward memory and cannot load the older checkpoint directly.

`walking_env.py` keeps Walker2d's forward movement, health, and control effort rewards, and adds:

- A penalty of 1 per unused foot per simulation step after that foot has not touched the floor for 1 second.
- A bonus up to 0.5 for changing the supporting foot, after 0.15 seconds of continuous single-foot support. The first support earns no bonus. The bonus is multiplied by forward velocity clipped to [0, 1], so standing or moving backward earns no switch bonus.

Tune `unused_after`, `unused_penalty`, `switch_bonus`, and `min_support_time` in `BothLegsReward` to change these incentives. They encourage both feet to participate but do not enforce a natural gait; contact tapping or shuffling may still emerge. Evaluate behavior in the viewer after training.

The viewer uses the same reward wrapper and observation shape as training. `both_legs.monitor.csv` records episode returns and the final step's reward components (not component totals). Model files, the virtual environment, and generated logs are ignored by Git.

## Joint strength limits

The XML caps commanded motor torque at +/-100 Nm for hips and knees and +/-10 Nm for ankles, using normalized controls in [-1, 1] and motor gears of 100 and 10. This is a deliberately weaker ankle setting, not a calibrated human muscle model. There are no separate toe joints, muscle fatigue, or muscle force-length/velocity dynamics. Passive forces and contact forces are not capped by the motor torque limit.

Change the two foot motor `gear` values in `walker2d_separated.xml` to tune ankle strength. Training and viewing share this XML. A separate checkpoint starts training fresh for the changed dynamics and preserves previous policies. Reduced ankle torque limits ankle-driven propulsion but does not guarantee that hopping disappears.

## Body weight

Total body mass is approximately 30 kg (reduced from 35 kg; original mass was 23.7 kg). Uniform geometry density scaling preserves the existing segment mass proportions and scales inertia consistently. Gravity is explicitly set to 9.81 m/s² downward. This is a simulation tuning experiment, not a human anatomical mass distribution. Hip/knee and ankle torque limits remain 100 and 10 Nm. Training uses a separate checkpoint for this body. If a 35 kg checkpoint existed when this change was applied, it was copied to seed the 30 kg policy; otherwise training starts fresh. Increased mass reduces acceleration under the same motor torque, but does not change free-fall acceleration or guarantee a natural gait.
