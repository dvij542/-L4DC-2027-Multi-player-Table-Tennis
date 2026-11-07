"""Render Figure 1 using the supplied simulator and diagnostics.

The robot state and target racket poses are read from SkillEnv.
Table supports, a visible mesh net, and thin racket faces are display refinements.
Four outer landing targets come from nash_skills.skills; target 3 is their center. The three preparation
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
import xml.etree.ElementTree as ET

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
        for step in range(200):
            action = np.concatenate([
                policy.predict(build_obs(obs, info, player), deterministic=True)[0]
                for player in (1, 2)
            ])
            obs, _, done, truncated, info = env.step(action)
            mujoco.mj_forward(env.model, env.data)
            ball = env.data.geom("ball_geom").xpos.copy()
            if step > 55 and 0.1 < ball[0] < 2.9 and 0.72 < ball[2] < 1.45 and abs(ball[1]) < 0.7:
                racket_heights = [float(env.data.geom(name).xpos[2]) for name in ("racket", "racket_opp")]
                score = (-abs(ball[0] - 1.25) - 0.25 * abs(ball[1])
                         - 0.4 * abs(ball[2] - 0.95)
                         - 0.5 * sum(abs(z - 1.0) for z in racket_heights))
                if score > best_score:
                    best_score = score
                    best = {
                        "qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(),
                        "ctrl": env.data.ctrl.copy(), "body_pos": env.model.body_pos.copy(),
                        "body_quat": env.model.body_quat.copy(), "time": env.data.time,
                        "episode": episode, "step": step, "seed": 43 + episode,
                        "skill_profile": env.skill_profile,
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
    parser.add_argument("--skill-profile", default="aggressive", choices=["current", "moderate_aggressive", "aggressive"])
    args = parser.parse_args()
    source, snapshot, output = (p.resolve() for p in (args.sim_root, args.snapshot, args.output))
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(source)
    sys.path.insert(0, str(source))
    from nash_skills.env_wrapper import SkillEnv
    from nash_skills.skills import SKILL_NAMES, world_target_xy
    import mujoco_env_comp
    # Render a visible textured floor inside MuJoCo. This changes materials,
    # not the simulator's robot, table, racket, or collision geometry.
    scene = ET.fromstring(mujoco_env_comp.xml)
    assets = ET.SubElement(scene, "asset")
    ET.SubElement(assets, "texture", name="figure_floor_texture", type="2d", builtin="checker",
                  rgb1="0.77 0.79 0.81", rgb2="0.83 0.85 0.87", width="512", height="512")
    ET.SubElement(assets, "material", name="figure_floor_material", texture="figure_floor_texture",
                  texuniform="true", texrepeat="0.5 0.5", reflectance="0.0", specular="0.10", shininess="0.1")
    scene.find(".//geom[@name='floor']").set("material", "figure_floor_material")
    scene.find(".//geom[@name='floor']").set("size", "200 200 0.1")
    # Display-only table detailing: all geoms are static and non-colliding.
    world = scene.find("worldbody")
    def display_geom(name, kind, **attributes):
        return ET.SubElement(world, "geom", name="figure_" + name, type=kind,
                             contype="0", conaffinity="0", **attributes)

    table_geom = scene.find(".//geom[@name='table_top']")
    table_body = scene.find(".//body[@name='table']")
    table_position = np.fromstring(table_body.get("pos"), sep=" ")
    table_size = np.fromstring(table_geom.get("size"), sep=" ")
    table_top = table_position[2] + table_size[2]
    table_bottom = table_position[2] - table_size[2]
    table_x = table_position[0]
    for y in (-0.7525, 0.7525):
        display_geom(f"table_sideline_{y}", "box", pos=f"{table_x} {y} {table_top + 0.001}",
                     size="1.37 0.008 0.001", rgba="0.95 0.96 0.96 1")
    for dx in (-1.36, 1.36):
        display_geom(f"table_endline_{dx}", "box", pos=f"{table_x + dx} 0 {table_top + 0.001}",
                     size="0.008 0.7625 0.001", rgba="0.95 0.96 0.96 1")
    display_geom("table_centerline", "box", pos=f"{table_x} 0 {table_top + 0.001}",
                 size="1.37 0.003 0.001", rgba="0.95 0.96 0.96 1")
    for dx in (-0.95, 0.95):
        for y in (-0.55, 0.55):
            display_geom(f"table_leg_{dx}_{y}", "box",
                         pos=f"{table_x + dx} {y} {table_bottom / 2}",
                         size=f"0.032 0.032 {table_bottom / 2}", rgba="0.18 0.22 0.26 1")
            display_geom(f"table_foot_{dx}_{y}", "box",
                         pos=f"{table_x + dx} {y} 0.014", size="0.047 0.047 0.014",
                         rgba="0.10 0.12 0.14 1")
        display_geom(f"table_crossbrace_{dx}", "box", pos=f"{table_x + dx} 0 0.18",
                     size="0.022 0.55 0.022", rgba="0.23 0.28 0.32 1")
    display_geom("table_underframe", "box", pos=f"{table_x} 0 {table_bottom - 0.012}",
                 size="1.38 0.7725 0.012", rgba="0.16 0.21 0.24 1")
    # Fine mesh, edge bands, posts, and clamps replace the source's solid bar.
    scene.find(".//geom[@name='net_geom']").set("rgba", "0 0 0 0")
    net_top = table_top + 0.1524
    display_geom("net_backing", "box", pos=f"{table_x} 0 {(table_top + net_top) / 2}",
                 size="0.0008 0.78 0.0762", rgba="0.12 0.17 0.20 0.18")
    for k, y in enumerate(np.linspace(-0.78, 0.78, 53)):
        display_geom(f"net_vertical_{k}", "capsule", size="0.0008",
                     fromto=f"{table_x} {y} {table_top} {table_x} {y} {net_top}",
                     rgba="0.17 0.21 0.23 0.85")
    for k, z in enumerate(np.linspace(table_top, net_top, 9)):
        display_geom(f"net_horizontal_{k}", "capsule", size="0.0008",
                     fromto=f"{table_x} -0.78 {z} {table_x} 0.78 {z}",
                     rgba="0.17 0.21 0.23 0.85")
    display_geom("net_top_band", "capsule", size="0.006",
                 fromto=f"{table_x} -0.81 {net_top} {table_x} 0.81 {net_top}",
                 rgba="0.95 0.96 0.96 1")
    for y in (-0.80, 0.80):
        display_geom(f"net_post_{y}", "capsule", size="0.012",
                     fromto=f"{table_x} {y} {table_bottom - 0.015} {table_x} {y} {net_top + 0.02}",
                     rgba="0.26 0.30 0.32 1")
        display_geom(f"net_clamp_{y}", "box", pos=f"{table_x} {y} {table_bottom}",
                     size="0.04 0.027 0.03", rgba="0.21 0.25 0.28 1")
    # A small offset from the exact side view makes the physical net visible,
    # while keeping the camera near the center of the two-player scene.
    camera = scene.find(".//camera[@name='overview']")
    camera_eye = np.array([0.65, -4.8, 3.2])
    camera_forward = np.array([table_x, 0, -0.35]) - camera_eye
    camera_forward /= np.linalg.norm(camera_forward)
    camera_right = np.cross(camera_forward, [0, 0, 1])
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_right, camera_forward)
    camera.set("pos", " ".join(map(str, camera_eye)))
    camera.set("xyaxes", " ".join(map(str, np.r_[camera_right, camera_up])))
    mujoco_env_comp.xml = ET.tostring(scene, encoding="unicode")
    env = SkillEnv(reset_mode="ready", skill_profile=args.skill_profile)
    try:
        if not snapshot.exists():
            if checkpoint is None:
                parser.error("--checkpoint is required when --snapshot does not exist")
            capture_snapshot(env, source, checkpoint, snapshot)
        state = np.load(snapshot, allow_pickle=False)
        snapshot_profile = str(state["skill_profile"]) if "skill_profile" in state else "current"
        if snapshot_profile != args.skill_profile:
            raise ValueError("Snapshot profile does not match --skill-profile; capture a new snapshot for that profile")
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
        # The source uses thick cylinders for robust contact handling. Thin only
        # the displayed faces after rollout capture; do not re-simulate with them.
        for racket_name in ("racket", "racket_opp"):
            m.geom_size[m.geom(racket_name).id, 1] = 0.006
        for target_name in ("cylinder", "cylinder_opp"):
            m.geom_size[m.geom(target_name).id, 1] = 0.005
        # Directional key light and restrained fill preserve 3-D shape and
        # floor shadows instead of flattening the scene with ambient light.
        m.geom_rgba[m.geom("floor").id] = [1, 1, 1, 1]
        m.geom_rgba[m.geom("table_top").id] = [0.06, 0.28, 0.36, 1]
        m.vis.headlight.ambient[:] = 0.08
        m.vis.headlight.diffuse[:] = 0.25
        m.vis.headlight.specular[:] = 0.10
        m.vis.quality.shadowsize = 8192
        m.vis.quality.offsamples = 8
        m.stat.center[:] = [table_x, 0, 0.55]
        m.stat.extent = 6.0
        m.light_pos[:] = [[-1.5, -3.0, 6.0], [4.5, 3.0, 6.0]]
        m.light_dir[:] = [[0.35, 0.5, -1], [-0.4, -0.3, -1]]
        m.light_type[:] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        m.light_diffuse[:] = [[0.50, 0.50, 0.50], [0.17, 0.17, 0.17]]
        m.light_ambient[:] = 0.02
        m.light_castshadow[:] = [True, False]
        mujoco.mj_forward(m, d)
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
        source_targets = np.array([(*world_target_xy(1, skill, profile=args.skill_profile), top + 0.002)
                                   for skill in SKILL_NAMES])
        targets = source_targets.copy()
        targets[2] = targets[[0, 1, 3, 4]].mean(axis=0)
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
        label(r"Ball velocity $\mathbf{v}^{b}$", (750, 165), ball + 0.075 * velocity, gold)
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
            "scene": "supplied SkillEnv", "skill_profile": args.skill_profile,
            "floor": "MuJoCo checker material with directional cast shadows",
            "snapshot_skill_profile": snapshot_profile, "snapshot_episode": int(state["episode"]),
            "snapshot_step": int(state["step"]), "mujoco_version": mujoco.__version__,
            "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            "target_names": SKILL_NAMES, "return_targets": targets.tolist(),
            "source_return_targets": source_targets.tolist(),
            "center_target_adjusted_for_illustration": True,
            "display_refinements": ["four table legs and braces", "mesh net with posts and bands", "12 mm racket thickness"],
            "camera_position": eye.tolist(), "racket_display_thickness_m": 0.012,
            "preparation_examples": preps,
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
