# Figure 1: supplied simulator scene

Figure 1 is rendered through the supplied `SkillEnv.render(mode="rgb_array")` implementation and its centered `overview` camera. It replaces the earlier illustrative scene assembled from separate assets.

## Inputs

The user provided these source folders:

- `vis scripts 2`: `mujoco_env_comp.py`, `iiwa14_comp.xml`, `diagnostic_fixed_skill.py`, and `diagnostic_rendering.py`.
- `nash_skills`: `env_wrapper.py`, `skills.py`, and `__init__.py`.

The sources are external dependencies, not copied into this paper repository. Their SHA-256 hashes are recorded in `figure1_provenance.json`. Missing robot meshes and the `logs/best_model_tracker1/best_model.zip` controller checkpoint were obtained from `dvij542/ping-pong-game` at commit `37dcb05731c0f07f9132412eec4f440b19ac11cf`, which is the simulator repository referenced by the supplied evaluation scripts. Mesh license information is retained in `figure1-third-party-LICENSE.txt`.

## What is taken directly from the simulator

- Both KUKA robots, mounted rackets, table, and net geometry are unchanged.
- The actual joint and ball state is a captured PPO rollout frame. `figure1_snapshot.npz` preserves that state and the moving target-pose transforms.
- PPO observations use the exact `build_ppo_obs` function extracted from the supplied `diagnostic_fixed_skill.py`.
- The green translucent rackets are the scene's own `vis` / `vis_opp` target geometries.
- The five landing locations are obtained with `world_target_xy(1, skill, profile="current")` in the order given by `SKILL_NAMES`: deep left, short left, center safe, short right, deep right.
- The ball arrow uses the captured translational velocity. Its display length is scaled for readability, not a calibrated velocity scale.
- Dashed blue areas show the XY command limits extracted from `KukaTennisEnv.step()`: local x in [-1.2, -0.3] m and local y in [-1, 1] m, mirrored through the second robot's root transform. These are the command clamps, not the different default MJCF joint limits.

The supplied scene represents gantries as sliding robot bases; it has **no gantry rail/frame meshes**. No fictitious hardware, table legs, or new robot geometry is added. Three preparation markers are schematic explanatory examples. The source defines one ready gantry position but no three-position preparation policy. Its ready x coordinate is retained and lateral examples y = -0.65, 0, +0.65 m are used. Their status is explicit in the image, caption, and metadata.

Background, lighting, and table color are adjusted only for print readability. All annotation anchors are projected through the camera actually used by MuJoCo.

## Rebuild from the saved frame

Validated with Python 3.12 and MuJoCo 4.0.0 on macOS. Assemble an external simulator directory with this structure:

```text
simulator/
  mujoco_env_comp.py
  iiwa14_comp.xml
  diagnostic_fixed_skill.py
  diagnostic_rendering.py
  assets/                  # robot OBJ meshes from ping-pong-game
  nash_skills/
    __init__.py
    env_wrapper.py
    skills.py
```

From the paper repository:

```bash
python3 -m venv work/figure1-venv
work/figure1-venv/bin/pip install 'mujoco==4.0.0' numpy scipy matplotlib Pillow gymnasium
work/figure1-venv/bin/python scripts/render_figure1.py \
  --sim-root /path/to/simulator \
  --snapshot scripts/figure1_snapshot.npz \
  --output Figures/figure1.png
```

This exports PNG, PDF with vector annotations, and JSON provenance. A raw RGB render is saved beside the snapshot. An offscreen OpenGL context is required.

To capture a new frame, additionally install `stable-baselines3` and `torch`, select a new snapshot path, and pass `--checkpoint /path/to/logs/best_model_tracker1/best_model.zip`. The script evaluates three short rollouts and selects a frame with the ball visible over the table; it does not retrain or evaluate strategic-method performance.
