# Figure 1: supplied simulator scene

Figure 1 is rendered through the supplied `SkillEnv.render(mode="rgb_array")` implementation and a near-central `overview` camera with a small lateral offset so the mesh net is visible. It replaces the earlier illustrative scene assembled from separate assets.

## Inputs

The user provided these source folders:

- `vis scripts 2`: `mujoco_env_comp.py`, `iiwa14_comp.xml`, `diagnostic_fixed_skill.py`, and `diagnostic_rendering.py`.
- `nash_skills`: `env_wrapper.py`, `skills.py`, and `__init__.py`.

The sources are external dependencies, not copied into this paper repository. Their SHA-256 hashes are recorded in `figure1_provenance.json`. Missing robot meshes and the `logs/best_model_tracker1/best_model.zip` controller checkpoint were obtained from `dvij542/ping-pong-game` at commit `37dcb05731c0f07f9132412eec4f440b19ac11cf`, which is the simulator repository referenced by the supplied evaluation scripts. Mesh license information is retained in `figure1-third-party-LICENSE.txt`.

## What is taken directly from the simulator

- Both KUKA robot meshes and the captured arm joint angles are preserved. For the approach illustration, each gantry is shifted sideways along its local y axis by 0.42 m within its command limits, with its longitudinal x coordinate unchanged while the target racket poses stay fixed. Table supports, net detailing, racket thickness, and the display ball size are refined as described below.
- `figure1_snapshot.npz` preserves the original PPO rollout state and moving target-pose transforms. The original file is retained unchanged; the renderer applies documented display offsets afterward. Short dashed green arrows point from the displayed rackets toward their targets.
- PPO observations use the exact `build_ppo_obs` function extracted from the supplied `diagnostic_fixed_skill.py`.
- The green translucent rackets are the scene's own `vis` / `vis_opp` target geometries.
- Source skill target coordinates are read and retained in the provenance JSON. For the illustration, targets 1, 2, 4, and 5 are placed 0.14 m inside the depth corners and 0.10 m inside the lateral edges of the right half: x = 1.64 or 2.73 m, y = +/-0.6625 m. Target 3 is their exact geometric center, (2.185, 0.0) m. These are display choices, separate from the source skill definitions, captured rollout, and reported results.
- The ball is a native MuJoCo sphere with specular shading and a cast shadow. Its native radius of 0.02 m is retained, giving a 40 mm diameter. There is no flat circle covering its shaded surface. The velocity arrow begins outside the sphere and uses the captured velocity with a 0.05 s display scale, shortened from 0.14 s; it is not a calibrated velocity scale.
- Dashed blue areas show the XY command limits extracted from `KukaTennisEnv.step()`: local x in [-1.2, -0.3] m and local y in [-1, 1] m, mirrored through the second robot's root transform. These are the command clamps, not the different default MJCF joint limits.

The supplied scene represents gantries as sliding robot bases; it has **no gantry rail/frame meshes**. For presentation, four table legs with feet and crossbraces, an underside frame, white tabletop lines, and a mesh net with posts, clamps, and a top band are added as static non-colliding display geometry. The source net bar is hidden. Mounted racket faces are displayed at 12 mm total thickness and target-pose rackets at 10 mm. These appearance changes are applied after the original rollout is captured; there is no re-simulation with thinner contact geometry. Three preparation markers are schematic explanatory examples. The source defines one ready gantry position but no three-position preparation policy. Its ready x coordinate is retained and lateral examples y = -0.65, 0, +0.65 m are used. Their status is explicit in the image, caption, and metadata.

A checker material is applied to the existing floor plane. A directional key light produces actual MuJoCo cast shadows; a weaker fill and low ambient light retain shading on the robot meshes. Table color is adjusted for contrast. The floor material, lighting, camera, added supports/net, and racket display sizes are presentation refinements, separate from the original simulation used to capture the state. All annotation anchors are projected through the camera actually used by MuJoCo.

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
  --skill-profile aggressive \
  --output Figures/figure1.png
```

This exports PNG, PDF with vector annotations, and JSON provenance. A raw RGB render is saved beside the snapshot. An offscreen OpenGL context is required.

To capture a new frame, additionally install `stable-baselines3` and `torch`, select a new snapshot path, and pass `--checkpoint /path/to/logs/best_model_tracker1/best_model.zip`. The script evaluates three short rollouts under the requested profile and selects a frame with the ball visible over the table; it does not retrain or evaluate strategic-method performance.
