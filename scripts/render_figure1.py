"""Render Figure 1 using the supplied simulator and diagnostics.

Scene geometry, target racket poses, and state are read from SkillEnv.
Five landing targets come from nash_skills.skills. The three preparation
markers are explanatory examples, because that set is absent from the source.
See figure1_README.md for staging the external simulator and rebuilding.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
import mujoco
import numpy as np
from PIL import Image


def supplied_ppo_observation(source: Path):
    """Use the diagnostics' exact observation function without importing eval CLI."""
    path = source / "diagnostic_fixed_skill.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_ppo_obs")
    namespace = {"np": np, "HISTORY": 4}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["build_ppo_obs"]


def capture_snapshot(env, source: Path, checkpoint: Path, destination: Path):
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(1)
    np.random.seed(43)
    policy = PPO.load(checkpoint, device="cpu", custom_objects={
        "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.2,
    })
    build_obs = supplied_ppo_observation(source)
    best, best_score = None, -np.inf
    for episode in range(3):
        env.set_skills("left_short", "right")
        obs, info = env.reset(seed=43 + episode)
        for step in range(170):
            action = np.concatenate([
                policy.predict(build_obs(obs, info, player), deterministic=True)[0]
                for player in (1, 2)
            ])
            obs, _, done, truncated, info = env.step(action)
            mujoco.mj_forward(env.model, env.data)
            ball = env.data.geom("ball_geom").xpos.copy()
            if step > 10 and 0.1 < ball[0] < 2.9 and 0.72 < ball[2] < 1.45 and abs(ball[1]) < 0.7:
                score = -abs(ball[0] - 1.25) - 0.25 * abs(ball[1]) - 0.4 * abs(ball[2] - 0.95)
                if score > best_score:
                    best_score = score
                    best = {
                        "qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(),
                        "ctrl": env.data.ctrl.copy(), "body_pos": env.model.body_pos.copy(),
                        "body_quat": env.model.body_quat.copy(), "time": env.data.time,
                        "episode": episode, "step": step, "seed": 43 + episode,
                    }
            if done or truncated:
                break
    if best is None:
        raise RuntimeError("No unobstructed mid-rally state was found")
    np.savez(destination, **best)


def commanded_gantry_limits(source: Path):
    """Read the commanded travel envelope directly from the supplied step()."""
    tree = ast.parse((source / "mujoco_env_comp.py").read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KukaTennisEnv")
    step = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    bounds = []
    for node in ast.walk(step):
        if (isinstance(node, ast.Call) and ast.unparse(node.func) == "np.clip"
                and ast.unparse(node.args[0]) in ("self.data.qpos[0]", "self.data.qpos[1]")):
            bounds.append((ast.unparse(node.args[0]), [ast.literal_eval(n) for n in node.args[1:3]]))
    limits = dict(bounds)
    return limits["self.data.qpos[0]"], limits["self.data.qpos[1]"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, snapshot, output = (p.resolve() for p in (args.sim_root, args.snapshot, args.output))
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(source)
    sys.path.insert(0, str(source))
    from nash_skills.env_wrapper import SkillEnv
    from nash_skills.skills import SKILL_NAMES, world_target_xy

    env = SkillEnv(reset_mode="ready", skill_profile="current")
    try:
        if not snapshot.exists():
            if checkpoint is None:
                parser.error("--checkpoint is required when --snapshot does not exist")
            capture_snapshot(env, source, checkpoint, snapshot)
        state = np.load(snapshot, allow_pickle=False)
        m, d = env.model, env.data
        d.qpos[:] = state["qpos"]
        d.qvel[:] = state["qvel"]
        d.ctrl[:] = state["ctrl"]
        d.time = float(state["time"])
        m.body_pos[:] = state["body_pos"]
        m.body_quat[:] = state["body_quat"]
        mujoco.mj_forward(m, d)
        width, height = 2400, 1400
        m.vis.global_.offwidth, m.vis.global_.offheight = width, height
        # Presentation only: printable background and balanced lighting.
        # No robot, racket, table, or net dimensions/transforms are modified.
        m.geom_rgba[m.geom("floor").id] = [0.98, 0.98, 0.98, 1]
        m.geom_rgba[m.geom("table_top").id] = [0.05, 0.24, 0.48, 1]
        m.vis.headlight.ambient[:] = 0.70
        m.vis.headlight.diffuse[:] = 0.55
        m.vis.headlight.specular[:] = 0.1
        m.light_diffuse[:] = 0.35
        m.light_ambient[:] = 0.08
        env._env._rgb_renderer = mujoco.Renderer(m, height=height, width=width)
        # Use the supplied RGB rendering path and its centered overview camera.
        image = env.render(mode="rgb_array")
        Image.fromarray(image).save(snapshot.parent / "raw-source-scene.png")
        camera = env._env._rgb_renderer.scene.camera
        eye = (np.array(camera[0].pos) + np.array(camera[1].pos)) / 2
        forward = np.array(camera[0].forward)
        up = np.array(camera[0].up)
        right = np.cross(forward, up)
        fovy = m.cam_fovy[m.camera("overview").id]
        focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))

        def project(point):
            p = np.asarray(point) - eye
            return np.array([width / 2 + focal * np.dot(p, right) / np.dot(p, forward),
                             height / 2 - focal * np.dot(p, up) / np.dot(p, forward)])

        fig = plt.figure(figsize=(12, 5.5), dpi=240)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(image)
        ax.set(xlim=(0, width), ylim=(1080, -20))
        ax.axis("off")
        ink, blue, green, gold = "#263440", "#20709a", "#237a61", "#b47a10"
        xlim, ylim = commanded_gantry_limits(source)
        # Source root transformations, not hand-built gantry hardware.
        for name in ("base", "base_opp"):
            body_id = m.body(name).id
            root = m.body_pos[body_id]
            rotation = np.eye(3) if name == "base" else np.diag([-1, -1, 1])
            corners = [root + rotation @ [x, y, 0.006]
                       for x, y in [(xlim[0], ylim[0]), (xlim[1], ylim[0]),
                                    (xlim[1], ylim[1]), (xlim[0], ylim[1])]]
            polygon = np.array([project(p) for p in corners])
            ax.add_patch(Polygon(polygon, closed=True, fc=blue, alpha=0.065, ec="none"))
            ax.add_patch(Polygon(polygon, closed=True, fill=False, ec=blue, lw=1.5, ls=(0, (5, 3))))
        top = float(d.geom("table_top").xpos[2] + m.geom_size[m.geom("table_top").id, 2])
        targets = [(*world_target_xy(1, skill), top + 0.002) for skill in SKILL_NAMES]
        for index, p in enumerate(targets, 1):
            xy = project(p)
            ax.add_patch(Circle(xy, 17, fc="#f3c45e", ec="#895b0d", lw=1, zorder=8))
            ax.text(*xy, str(index), fontsize=9, weight="bold", ha="center", va="center", color="#4c3509", zorder=9)
        # Only the center ready position is defined in the supplied source.
        # Outer lateral positions are schematic examples, explicitly captioned.
        from mujoco_env_comp import READY_GANTRY_X
        prep_x = float(m.body_pos[m.body("base_opp").id, 0] - READY_GANTRY_X)
        preps = [(prep_x, y, 0.015) for y in (-0.65, 0.0, 0.65)]
        for index, p in enumerate(preps, 1):
            xy = project(p)
            ax.add_patch(Circle(xy, 23, fc="#edf7f1", ec=green, lw=1.4, zorder=8))
            ax.text(*xy, rf"$m_{index}$", fontsize=10, ha="center", va="center", color=green, zorder=9)
        ball = d.geom("ball_geom").xpos.copy()
        velocity = d.qvel[m.jnt_dofadr[m.joint("haha").id]:][:3].copy()
        velocity_tip = ball + 0.14 * velocity
        ax.add_patch(FancyArrowPatch(project(ball), project(velocity_tip), arrowstyle="-|>",
                                    mutation_scale=18, color="#d18110", lw=2, zorder=11))
        ax.add_patch(Circle(project(ball), 6, fc="#ffb02e", ec="#78530b", lw=0.7, zorder=12))

        def label(text, at, point, color=ink):
            ax.annotate(text, xy=project(point), xytext=at, ha="center", va="center", fontsize=13,
                        color=color, bbox=dict(boxstyle="round,pad=.25", fc="white", ec="none", alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color=color, lw=1.1), zorder=20)

        label("KUKA iiwa + mounted racket", (340, 55), d.geom("racket").xpos)
        label("KUKA iiwa + mounted racket", (2050, 55), d.geom("racket_opp").xpos)
        label("Target racket pose", (355, 470), d.geom("cylinder").xpos, green)
        label("Target racket pose", (2050, 470), d.geom("cylinder_opp").xpos, green)
        label("Ball", (985, 100), ball, gold)
        label(r"Ball velocity $\mathbf{v}^{b}$", (1360, 80), ball + 0.075 * velocity, gold)
        label("Gantry movable region", (335, 1010), [-1.15, -0.8, 0.006], blue)
        label("Gantry movable region", (2060, 1010), [4.15, -0.8, 0.006], blue)
        label(r"Return choices $k_1,\ldots,k_5$", (1370, 825), [2.30, -0.10, top + 0.002], gold)
        label("Preparation positions\n(schematic)", (2000, 820), preps[0], green)
        fig.savefig(output, dpi=240, facecolor="white", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        source_files = [source / n for n in ["mujoco_env_comp.py", "iiwa14_comp.xml", "diagnostic_rendering.py",
                        "diagnostic_fixed_skill.py", "nash_skills/skills.py", "nash_skills/env_wrapper.py"]]
        metadata = {
            "scene": "supplied SkillEnv", "skill_profile": "current", "snapshot_episode": int(state["episode"]),
            "snapshot_step": int(state["step"]), "mujoco_version": mujoco.__version__,
            "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            "target_names": SKILL_NAMES, "return_targets": targets, "preparation_examples": preps,
            "preparation_markers_are_schematic": True, "commanded_gantry_limits": [xlim, ylim],
            "ball_position": ball.tolist(), "ball_velocity": velocity.tolist(),
            "velocity_display_time_scale": 0.14, "camera_name": "overview",
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        print("Saved", output, "using supplied SkillEnv.render(mode='rgb_array')")
    finally:
        env.close()


if __name__ == "__main__":
    main()
