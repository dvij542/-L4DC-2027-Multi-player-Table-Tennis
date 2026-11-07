# Figure 1

`render_figure1.py` renders a real MuJoCo scene, then projects vector annotations through MuJoCo's render camera. The output PDF contains the rendered scene as a raster image and editable vector text/markers.

This is an **illustrative layout**, not a captured experimental state. The custom XY gantries, their travel envelopes, target locations, preparation positions, racket target poses, ball state, and camera are chosen for explanation. They must be matched to the experimental environment before being described as its exact geometry. The illustration is not evidence of controller performance.

## Source and license

The KUKA iiwa MJCF and meshes come from `MujocoRecon` in:

https://github.com/sastry-group/LATTE-MV/tree/bcada1f744af3e1c39fd5adb5ee5bb534f0d1ba2/MujocoRecon

The model's accompanying BSD 3-Clause license is retained in `figure1-third-party-LICENSE.txt` (copyright 2022, Max Planck Gesellschaft). The rendering script duplicates the arm, adds schematic gantry rails, replaces the racket geometry, and chooses display poses by inverse kinematics. External meshes are not vendored in this paper repository.

## Rebuild

Validated with Python 3.12 and MuJoCo 4.0.0 on macOS. From the paper repository:

```bash
python3 -m venv work/figure1-venv
work/figure1-venv/bin/pip install 'mujoco==4.0.0' numpy scipy matplotlib Pillow
git clone https://github.com/sastry-group/LATTE-MV.git work/LATTE-MV
git -C work/LATTE-MV checkout bcada1f744af3e1c39fd5adb5ee5bb534f0d1ba2
work/figure1-venv/bin/python scripts/render_figure1.py \
  --source work/LATTE-MV/MujocoRecon \
  --output Figures/figure1.png \
  --workdir work/figure1
```

The script exports `figure1.png`, `figure1.pdf`, the intermediate MJCF, an unannotated render, and a JSON record of the illustrated coordinates, joint poses, and ball state. An offscreen OpenGL context is required. On Linux, configure an appropriate MuJoCo rendering backend.

The camera is centered on the table, with both gantries visible. Blue outlines indicate the gantry travel envelopes; dashed red rackets are target poses; yellow numbered markers indicate five candidate return locations on the right half; green markers denote three right-robot preparation positions. The orange arrow shows the illustrated ball velocity direction; its display length is not a calibrated velocity scale.
