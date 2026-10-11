# Two-legged walker

Use the project virtual environment:

```sh
.venv/bin/python train.py --timesteps 1000000
.venv/bin/python watch.py
.venv/bin/python -m unittest test_walking_env.py
```

Training resumes `walker2d_both_legs_30kg_torso15_ankle10.zip` if present, otherwise starts a fresh PPO policy. Older checkpoints are preserved. The new policy has five extra observations for reward memory and cannot load the older checkpoint directly.

`walking_env.py` keeps Walker2d's forward movement, health, and control effort rewards, and adds:

- A penalty of 1 per unused foot per simulation step after that foot has not touched the floor for 1 second.
- A bonus up to 0.5 for changing the supporting foot, after 0.15 seconds of continuous single-foot support. The first support earns no bonus. The bonus is multiplied by forward velocity clipped to [0, 1], so standing or moving backward earns no switch bonus.

Tune `unused_after`, `unused_penalty`, `switch_bonus`, and `min_support_time` in `BothLegsReward` to change these incentives. They encourage both feet to participate but do not enforce a natural gait; contact tapping or shuffling may still emerge. Evaluate behavior in the viewer after training.

The viewer uses the same reward wrapper and observation shape as training. `both_legs.monitor.csv` records episode returns and the final step's reward components (not component totals). Model files, the virtual environment, and generated logs are ignored by Git.

## Joint strength limits

The XML caps commanded motor torque at +/-100 Nm for hips and knees and +/-10 Nm for ankles, using normalized controls in [-1, 1] and motor gears of 100 and 10. This is a deliberately weaker ankle setting, not a calibrated human muscle model. There are no separate toe joints, muscle fatigue, or muscle force-length/velocity dynamics. Passive forces and contact forces are not capped by the motor torque limit.

Change the two foot motor `gear` values in `walker2d_separated.xml` to tune ankle strength. Training and viewing share this XML. A separate checkpoint starts training fresh for the changed dynamics and preserves previous policies. Reduced ankle torque limits ankle-driven propulsion but does not guarantee that hopping disappears.

## Body weight

Total body mass is 30 kg: torso 15 kg, each thigh 4 kg, each lower leg 2.5 kg, and each foot 1 kg. Explicit geometry masses replace the previous uniform-density distribution. MuJoCo infers each segment's inertia from its geometry and assigned mass. Gravity remains 9.81 m/s² downward; hip/knee and ankle torque limits remain 100 and 10 Nm. This is a simplified tuning experiment, not a calibrated human anatomical model.

The torso-weighted body has its own checkpoint. If the previous 30 kg policy exists when this change is applied, it is copied to seed the new policy while preserving the original. Further training is needed to adapt to changed dynamics. Redistributing mass does not guarantee a natural gait.
