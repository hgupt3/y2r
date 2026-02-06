"""Test URDF → USD conversion and inspect the UR5e + LEAP Hand robot.

Usage:
    # Headless: print joint/body info only
    python isaac_scripts/generate_usd.py

    # Visual: open livestream viewer with debug axes
    python isaac_scripts/generate_usd.py --view

Viewer legend (--view mode):
    palm_link  — dim RGB axes (R=X, G=Y, B=Z)
    palm_frame — bright RGB axes
    camera     — bright axes with CAMERA convention:
                   Red   = +X = camera forward (optical +Z)
                   Green = +Y = camera left    (optical -X)
                   Blue  = +Z = camera up      (optical -Y)
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test UR5e + LEAP Hand URDF conversion.")
parser.add_argument("--view", action="store_true", help="Open livestream viewer to visually inspect.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.view:
    args.livestream = 2

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply
from isaaclab_assets.robots import UR5E_LEAP_CFG

# Create simulation
sim_cfg = sim_utils.SimulationCfg(dt=1 / 120.0, device="cpu")
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view(eye=(1.5, 1.5, 1.0), target=(0.0, 0.0, 0.3))

# Spawn the robot
robot_cfg = UR5E_LEAP_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)

# Initialize physics
sim.reset()

# Apply initial joint positions
robot.reset()

print("\n" + "=" * 60)
print("UR5e + LEAP Hand — URDF Conversion Report")
print("=" * 60)

print(f"\nNumber of joints:  {robot.num_joints}")
print(f"Number of bodies:  {robot.num_bodies}")

print("\n--- Joint names ---")
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name}")

print("\n--- Body names ---")
for i, name in enumerate(robot.body_names):
    print(f"  [{i:2d}] {name}")

print("\n--- Joint limits (degrees) ---")
limits = robot.root_physx_view.get_dof_limits().squeeze(0)
for i, name in enumerate(robot.joint_names):
    lo = torch.rad2deg(limits[i, 0]).item()
    hi = torch.rad2deg(limits[i, 1]).item()
    print(f"  {name:25s}  [{lo:8.2f}, {hi:8.2f}]")

print("\n--- Initial joint positions (degrees) ---")
pos = robot.data.joint_pos.squeeze(0)
for i, name in enumerate(robot.joint_names):
    deg = torch.rad2deg(pos[i]).item()
    print(f"  {name:25s}  {deg:8.2f}")

print("\n--- USD Prim Hierarchy ---")
stage = sim.stage
root_prim = stage.GetPrimAtPath("/World/Robot")
def print_tree(prim, indent=0):
    typ = prim.GetTypeName()
    if typ in ("Xform", ""):
        print(f"  {'  ' * indent}{prim.GetName()}  -> {prim.GetPath()}")
        for child in prim.GetChildren():
            print_tree(child, indent + 1)
print_tree(root_prim)

# =============================================================================
# Frame poses — print world poses for key frames
# =============================================================================

# Helper: look up a body by name, returns (index, found) — safe for optional bodies
def find_body(name):
    ids = robot.find_bodies(name)[0]
    if len(ids) == 0:
        return None
    return ids[0]

robot.update(sim.cfg.dt)

# Bodies to report (name → index); None means body not found in this URDF
REPORT_BODIES = {
    "palm_link": find_body("palm_link"),
    "palm_frame": find_body("palm_frame"),
    "gemini_305_left_camera_optical_frame": find_body("gemini_305_left_camera_optical_frame"),
    "gemini_305_link": find_body("gemini_305_link"),
}

# Unit vectors for axis computation
X_AXIS = torch.tensor([1.0, 0.0, 0.0])
Y_AXIS = torch.tensor([0.0, 1.0, 0.0])
Z_AXIS = torch.tensor([0.0, 0.0, 1.0])

def quat_apply_single(quat, vec):
    """Apply quaternion rotation to a single vector."""
    return quat_apply(quat.unsqueeze(0), vec.unsqueeze(0)).squeeze(0)

print("\n--- Frame poses (world) ---")
for name, idx in REPORT_BODIES.items():
    if idx is None:
        print(f"\n  {name}: NOT FOUND")
        continue
    body_pos = robot.data.body_pos_w[0, idx]
    body_quat = robot.data.body_quat_w[0, idx]
    print(f"\n  {name}:")
    print(f"    pos:  ({body_pos[0]:.5f}, {body_pos[1]:.5f}, {body_pos[2]:.5f})")
    print(f"    quat: ({body_quat[0]:.5f}, {body_quat[1]:.5f}, {body_quat[2]:.5f}, {body_quat[3]:.5f})  [wxyz]")

    # For camera frames, print raw axis directions + camera convention
    if "gemini_305" in name:
        opt_x = quat_apply_single(body_quat, X_AXIS)
        opt_y = quat_apply_single(body_quat, Y_AXIS)
        opt_z = quat_apply_single(body_quat, Z_AXIS)
        print(f"    +X (world): ({opt_x[0]:.4f}, {opt_x[1]:.4f}, {opt_x[2]:.4f})")
        print(f"    +Y (world): ({opt_y[0]:.4f}, {opt_y[1]:.4f}, {opt_y[2]:.4f})")
        print(f"    +Z (world): ({opt_z[0]:.4f}, {opt_z[1]:.4f}, {opt_z[2]:.4f})")
        if "optical" in name:
            # ROS optical: X=right, Y=down, Z=forward
            # User camera convention: X=forward, Y=left, Z=up
            cam_forward = quat_apply_single(body_quat, Z_AXIS)    # optical +Z
            cam_left = quat_apply_single(body_quat, -X_AXIS)      # optical -X
            cam_up = quat_apply_single(body_quat, -Y_AXIS)        # optical -Y
            print(f"    camera forward: ({cam_forward[0]:.4f}, {cam_forward[1]:.4f}, {cam_forward[2]:.4f})  [optical +Z]")
            print(f"    camera left:    ({cam_left[0]:.4f}, {cam_left[1]:.4f}, {cam_left[2]:.4f})  [optical -X]")
            print(f"    camera up:      ({cam_up[0]:.4f}, {cam_up[1]:.4f}, {cam_up[2]:.4f})  [optical -Y]")

print("\n" + "=" * 60)
print("Conversion successful!")
print("=" * 60)

# =============================================================================
# Visual debug draw (--view mode)
# =============================================================================

if args.view:
    # Initialize debug draw
    debug_draw = None
    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    except ModuleNotFoundError:
        try:
            from omni.isaac.debug_draw import _debug_draw as omni_debug_draw
        except ModuleNotFoundError:
            from omni.debugdraw import get_debug_draw_interface
            debug_draw = get_debug_draw_interface()
            omni_debug_draw = None
    if omni_debug_draw is not None:
        debug_draw = omni_debug_draw.acquire_debug_draw_interface()

    # -- Drawing config --
    # Standard RGB axes for body frames
    FRAME_AXIS_LEN = 0.12  # 12cm
    # Camera axes (slightly longer for visibility)
    CAMERA_AXIS_LEN = 0.15  # 15cm

    # Indices (pre-looked-up)
    palm_link_idx = REPORT_BODIES["palm_link"]
    palm_frame_idx = REPORT_BODIES["palm_frame"]
    camera_optical_idx = REPORT_BODIES["gemini_305_left_camera_optical_frame"]

    print("\nViewer running at http://localhost:8211")
    print("Frame axes:")
    print("  palm_link  = dim   RGB  (R=X, G=Y, B=Z)")
    print("  palm_frame = bright RGB  (R=X, G=Y, B=Z)")
    if camera_optical_idx is not None:
        print("  camera     = bright, CAMERA convention:")
        print("               Red=forward(+X)  Green=left(+Y)  Blue=up(+Z)")
    print("Press Ctrl+C to exit.\n")

    def draw_body_axes(body_idx, axis_len, line_width, dim=False):
        """Append RGB axis lines for a body frame to the draw lists.

        Args:
            body_idx: Index into robot.data.body_pos_w / body_quat_w
            axis_len: Length of each axis line (meters)
            line_width: Pixel width of lines
            dim: If True, use half-brightness colors
        """
        body_pos = robot.data.body_pos_w[0, body_idx]
        body_quat = robot.data.body_quat_w[0, body_idx]
        brightness = 0.5 if dim else 1.0
        alpha = 0.7 if dim else 1.0
        for axis, rgb in [(X_AXIS, (brightness, 0.0, 0.0)),
                          (Y_AXIS, (0.0, brightness, 0.0)),
                          (Z_AXIS, (0.0, 0.0, brightness))]:
            end = body_pos + quat_apply_single(body_quat, axis) * axis_len
            starts.append(body_pos.tolist())
            ends.append(end.tolist())
            colors.append((*rgb, alpha))
            sizes.append(line_width)

    def draw_camera_axes(body_idx, axis_len, line_width):
        """Draw camera-convention axes for the optical frame.

        Maps ROS optical frame (X=right, Y=down, Z=forward) to camera convention:
            Red   (+X, forward) = optical +Z
            Green (+Y, left)    = optical -X
            Blue  (+Z, up)      = optical -Y
        """
        body_pos = robot.data.body_pos_w[0, body_idx]
        body_quat = robot.data.body_quat_w[0, body_idx]
        cam_axes = [
            (Z_AXIS,  (1.0, 0.0, 0.0, 1.0)),   # forward (+Z_opt) → Red
            (-X_AXIS, (0.0, 1.0, 0.0, 1.0)),    # left (-X_opt) → Green
            (-Y_AXIS, (0.0, 0.0, 1.0, 1.0)),    # up (-Y_opt) → Blue
        ]
        for axis_local, color in cam_axes:
            end = body_pos + quat_apply_single(body_quat, axis_local) * axis_len
            starts.append(body_pos.tolist())
            ends.append(end.tolist())
            colors.append(color)
            sizes.append(line_width)

    while simulation_app.is_running():
        sim.step()
        robot.update(sim.cfg.dt)

        if debug_draw is not None:
            debug_draw.clear_lines()
            starts = []
            ends = []
            colors = []
            sizes = []

            # palm_link (dim RGB axes)
            if palm_link_idx is not None:
                draw_body_axes(palm_link_idx, FRAME_AXIS_LEN, 3.0, dim=True)

            # palm_frame (bright RGB axes)
            if palm_frame_idx is not None:
                draw_body_axes(palm_frame_idx, FRAME_AXIS_LEN, 5.0, dim=False)

            # Camera optical frame (camera convention axes)
            if camera_optical_idx is not None:
                draw_camera_axes(camera_optical_idx, CAMERA_AXIS_LEN, 5.0)

            debug_draw.draw_lines(starts, ends, colors, sizes)
else:
    simulation_app.close()
