"""Test URDF → USD conversion and inspect the UR5e + LEAP Hand robot."""
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

# --- Frame axis visualization (--view mode) ---
# Draw RGB axes on palm_link (dim) and palm_frame (bright) so you can
# visually compare them and tweak the URDF palm_frame_joint transform.

# Find body indices
palm_link_idx = robot.find_bodies("palm_link")[0][0]
palm_frame_ids = robot.find_bodies("palm_frame")[0]
palm_frame_idx = palm_frame_ids[0] if len(palm_frame_ids) > 0 else None

# Print current poses
robot.update(sim.cfg.dt)
pl_pos = robot.data.body_pos_w[0, palm_link_idx]
pl_quat = robot.data.body_quat_w[0, palm_link_idx]
print(f"\n--- palm_link pose (world) ---")
print(f"  pos: ({pl_pos[0]:.5f}, {pl_pos[1]:.5f}, {pl_pos[2]:.5f})")
print(f"  quat (wxyz): ({pl_quat[0]:.5f}, {pl_quat[1]:.5f}, {pl_quat[2]:.5f}, {pl_quat[3]:.5f})")

if palm_frame_idx is not None:
    pf_pos = robot.data.body_pos_w[0, palm_frame_idx]
    pf_quat = robot.data.body_quat_w[0, palm_frame_idx]
    print(f"\n--- palm_frame pose (world) ---")
    print(f"  pos: ({pf_pos[0]:.5f}, {pf_pos[1]:.5f}, {pf_pos[2]:.5f})")
    print(f"  quat (wxyz): ({pf_quat[0]:.5f}, {pf_quat[1]:.5f}, {pf_quat[2]:.5f}, {pf_quat[3]:.5f})")

print("\n" + "=" * 60)
print("Conversion successful!")
print("=" * 60)

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

    AXIS_LEN = 0.12  # 12cm axes
    x_axis = torch.tensor([1.0, 0.0, 0.0])
    y_axis = torch.tensor([0.0, 1.0, 0.0])
    z_axis = torch.tensor([0.0, 0.0, 1.0])

    print("\nViewer running at http://localhost:8211")
    print("Frame axes:  palm_link = dim,  palm_frame = bright")
    print("  Red=X  Green=Y  Blue=Z")
    print("Press Ctrl+C to exit.\n")

    while simulation_app.is_running():
        sim.step()
        robot.update(sim.cfg.dt)

        if debug_draw is not None:
            debug_draw.clear_lines()
            starts = []
            ends = []
            colors = []
            sizes = []

            # palm_link axes (dim colors, thinner)
            p_pos = robot.data.body_pos_w[0, palm_link_idx]
            p_quat = robot.data.body_quat_w[0, palm_link_idx]
            for axis, color in [(x_axis, (0.5, 0.0, 0.0, 0.7)),
                                (y_axis, (0.0, 0.5, 0.0, 0.7)),
                                (z_axis, (0.0, 0.0, 0.5, 0.7))]:
                end = p_pos + quat_apply(p_quat.unsqueeze(0), axis.unsqueeze(0)).squeeze(0) * AXIS_LEN
                starts.append(p_pos.tolist())
                ends.append(end.tolist())
                colors.append(color)
                sizes.append(3.0)

            # palm_frame axes (bright colors, thicker)
            if palm_frame_idx is not None:
                pf_pos = robot.data.body_pos_w[0, palm_frame_idx]
                pf_quat = robot.data.body_quat_w[0, palm_frame_idx]
                for axis, color in [(x_axis, (1.0, 0.0, 0.0, 1.0)),
                                    (y_axis, (0.0, 1.0, 0.0, 1.0)),
                                    (z_axis, (0.0, 0.0, 1.0, 1.0))]:
                    end = pf_pos + quat_apply(pf_quat.unsqueeze(0), axis.unsqueeze(0)).squeeze(0) * AXIS_LEN
                    starts.append(pf_pos.tolist())
                    ends.append(end.tolist())
                    colors.append(color)
                    sizes.append(5.0)

            debug_draw.draw_lines(starts, ends, colors, sizes)
else:
    simulation_app.close()
