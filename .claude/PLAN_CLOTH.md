# Cloth / Deformable Object Support Plan

Reference doc for adding cloth manipulation to the RL training pipeline.

---

## High-Level Architecture

**Per-GPU homogeneous envs. Multi-GPU combines at training level.**

```
GPU 0 (Learner + Rigid Actor)          GPU 1 (Cloth Actor)
┌──────────────────────────┐          ┌──────────────────────────┐
│  enable_cloth: false     │          │  enable_cloth: true      │
│  16k rigid envs          │          │  4k cloth envs           │
│  RigidObject as object   │          │  DeformableObject as obj │
│  Pose-based trajectory   │          │  Keypoint trajectory     │
│  Standard rewards        │          │  Keypoint rewards        │
│                          │          │                          │
│  Collects rigid rollouts │          │  Collects cloth rollouts │
│  + receives cloth data   │◄─────────│  Sends data to GPU 0     │
│  Trains PPO on combined  │──────────►  Receives updated weights│
│  Broadcasts new weights  │          │                          │
└──────────────────────────┘          └──────────────────────────┘
```

- `enable_cloth: true` makes the ENTIRE simulation cloth-only
- No mixed rigid+deformable in same scene
- No per-env masking needed
- Each GPU runs homogeneous envs
- Multi-GPU learner-worker architecture combines them for shared policy training

---

## Technical Foundation

### Isaac Lab DeformableObject API

```python
# Spawning
DeformableObjectCfg(
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.3, 0.3, 0.005),  # thin sheet
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            rest_offset=0.0, contact_offset=0.001,
        ),
        physics_material=sim_utils.DeformableBodyMaterialCfg(
            youngs_modulus=1e4,     # low = soft (cloth-like)
            poissons_ratio=0.45,    # high = volume-preserving
            dynamic_friction=0.5,
        ),
    ),
)

# Runtime data
cloth.data.nodal_pos_w          # (num_envs, max_vertices, 3) - every step
cloth.data.nodal_vel_w          # (num_envs, max_vertices, 3)
cloth.data.default_nodal_state_w # (num_envs, max_vertices, 6) - at spawn
cloth.data.root_pos_w           # (num_envs, 3) - mean of all nodes

# Reset (reset() is a no-op, must write manually)
cloth.write_nodal_state_to_sim(nodal_state, env_ids)
cloth.write_nodal_kinematic_target_to_sim(targets, env_ids)

# Node indices are stable: node N = same material point always (FEM property)
```

### Key Constraints
- GPU only (no CPU simulation for deformable)
- `replicate_physics: false` required
- Multi-GPU deformable bug (open): https://github.com/isaac-sim/IsaacLab/issues/3117
  - Workaround: CUDA_VISIBLE_DEVICES=N (each process sees single GPU, torchrun handles this)
- No `root_quat_w` on DeformableObject (no single rotation for a deforming body)
- Experimental API: "subject to change due to changes on underlying PhysX API"
- No real cloth in Isaac Lab — we approximate with thin FEM soft bodies

---

## Milestones

### M0: Proof of Life
**Goal**: Thin FEM body in scene, robot can interact, physics is stable.

**What to do:**
- Add `cloth.enabled` flag to config (in base.yaml, default false)
- When enabled, `trajectory_env_cfg.py` spawns `DeformableObjectCfg` as `scene.object`
  instead of `RigidObjectCfg`
- Disable all observation/reward/termination terms that assume rigid body
  (they access root_quat_w etc. which doesn't exist on DeformableObject)
- Tune material params until cloth-like behavior:
  - Young's modulus: ~1e3-1e4 Pa
  - Poisson's ratio: ~0.45
  - Dynamic friction: ~0.5
  - Thickness: ~0.003-0.01m
- Test with keyboard script: `./isaac_scripts/keyboard.sh --task cloth`
- Verify: robot hand can push/pinch/lift it, it deforms and drapes
- Verify: `cloth.data.nodal_pos_w` returns sensible per-node positions
- Measure: FPS with cloth vs rigid (determines how many cloth envs are feasible)

**What to build:**
- `configs/layers/tasks/cloth.yaml` — task layer enabling cloth
- Modifications to `trajectory_env_cfg.py` — conditional DeformableObjectCfg spawning
- Guard clauses in observation/reward/termination builders — skip rigid-only terms when cloth

**Risk**: If thin FEM bodies don't behave cloth-like enough or are unstable with
hand contact, the entire approach needs rethinking.

### M1: Observation Pipeline
**Goal**: Policy observes cloth state through point cloud interface, same shape as rigid.

**Representation change:**
```
Rigid:  points_local cached at init, transformed each step via quat_apply(quat, pts) + pos
Cloth:  selected node indices cached at init, read directly from nodal_pos_w each step
```

**What to do:**
- At init: identify K node indices from FEM mesh
  - Sort nodes by position at spawn time to find corners, edge midpoints, interior
  - Select 32 nodes (same count as rigid) spread across the surface
  - Cache selected indices
- At runtime: `cloth.data.nodal_pos_w[:, selected_indices, :]` → (num_envs, 32, 3)
  - Transform to robot base frame (same as rigid)
  - Flatten to (num_envs, 96) — same output shape
- Teacher: all 32 points
- Student visible point cloud: compute normals from neighboring nodes, back-face cull
  - Or simpler: cloth is thin, most points visible from any angle — maybe skip culling
- Target point clouds: same K keypoints at their target positions

**Key file**: `mdp/observations.py` — new `cloth_point_cloud_b` term or modify
`object_point_cloud_b` with cloth branch.

### M2: Cloth Trajectory Generation
**Goal**: Generate dense keypoint target sequences for cloth tasks.

**Representation:**
```
Rigid trajectory:  (num_envs, total_targets, 7)      — pos + quat per step
Cloth trajectory:  (num_envs, total_targets, K, 3)   — K keypoint positions per step
```

**Segments for cloth (simpler than rigid):**
```yaml
cloth_segments:
  - name: "approach"     # hand moves to cloth edge
    duration: 1.5
  - name: "pinch"        # fingers close on cloth
    duration: 1.0
  - name: "manipulate"   # drag/fold motion
    duration: 3.0
  - name: "release"      # fingers open, retreat
    duration: 1.5
```

**Task: Slide (simplest)**
- Cloth starts flat at position A on table
- Target: same flat shape at position B
- Keypoint trajectory: linear interpolation with easing for all K keypoints
- Hand: approach edge → pinch → drag → release
- Randomize: start position, slide direction, slide distance

**Task: Flatten**
- Cloth starts with mild Z perturbation on nodes
- Target: flat keypoint positions on table
- Hand: press down on raised areas
- Randomize: perturbation amplitude, which areas raised

**Task: Fold**
- Cloth starts flat
- Target: folded (one edge overlaps opposite edge)
- Dense intermediate targets showing fold arc at every timestep:
  ```
  t=0.0: flat, all keypoints at rest
  t=0.2: moving-side keypoints lifting off table
  t=0.5: moving-side at peak height, curling over fold axis
  t=0.8: moving-side descending onto stationary side
  t=1.0: fold complete
  ```
- Moving keypoints follow circular arc: x = r*cos(t*pi), z = r*sin(t*pi)
- Interior keypoints interpolate based on distance from fold axis
- Hand trajectory: pinch moving edge, follow the arc, release
- Randomize: fold axis (horiz/vert/diagonal), direction, which edge to grasp

**Hand trajectory for cloth:**
Unlike rigid (hand coupled to object via local-frame offset), cloth hand trajectory
is computed independently:
- Approach position: near the cloth edge to be grasped
- Pinch: hand at edge, fingers close
- Manipulate: hand follows a path (drag line or fold arc)
- Release: hand opens and retreats
The cloth keypoint targets tell us WHAT should happen. The hand targets tell the
policy HOW to make it happen.

**Key file**: `trajectory_manager.py` — new code path for cloth trajectory generation.
Could be a separate class `ClothTrajectoryManager` or a mode within existing manager.

### M3: Rewards & Terminations for Cloth
**Goal**: Policy receives learning signal for cloth manipulation.

**New/modified rewards:**
| Reward | Description | Replaces |
|--------|-------------|----------|
| Keypoint tracking | `exp(-mean_keypoint_dist / std)` | `lookahead_tracking` |
| Keypoint progress | Fractional improvement in mean keypoint error | `tracking_progress` |
| Hand following | Same as rigid — hand tracks its 7-DOF target | `hand_pose_following` (unchanged) |
| Contact reward | Thumb + finger contact with cloth surface | `fingers_to_object` |
| Keypoint success | Release phase: keypoints within threshold | `trajectory_success` |

**Keep unchanged**: `action_l2`, `action_rate_l2`, `arm_table_penalty`,
`early_termination`, `finger_regularizer`

**Drop for cloth**: `finger_manipulation` (no in-hand rotation), `object_stillness`
(cloth settles differently)

**Termination**: Mean keypoint error > threshold → terminate (replaces pose-based
`trajectory_deviation`)

**Key file**: `mdp/rewards.py` — cloth branches in existing reward functions or
new cloth-specific terms.

### M4: Curriculum & Domain Randomization
**Goal**: Robust training that generalizes across cloth properties.

**Material DR:**
- Young's modulus: range of stiffnesses [1e3, 1e5]
- Poisson's ratio: [0.3, 0.49]
- Dynamic friction: [0.2, 0.8]
- Density: light vs heavy fabric
- Cloth thickness: [0.003, 0.01]

**Size DR:**
- Cloth width/height: [0.2, 0.5]m

**Task curriculum:**
- Difficulty 0-3: slide only (easiest)
- Difficulty 4-6: add flatten
- Difficulty 7-8: add simple folds (horizontal/vertical)
- Difficulty 9-10: add diagonal folds, partial folds

**Other curriculum:**
- Gravity ramp (same as rigid)
- Tracking thresholds tighten
- Error gate tightens

### M5: Validation & Polish
**Goal**: Cloth training works end-to-end on single GPU.

- Play mode visualization of cloth manipulation
- Verify slide, flatten, fold tasks learn successfully
- Tune reward weights, curriculum pacing
- Student distillation: camera observations work for cloth
- Performance profiling: envs/sec, determine optimal cloth env count

### M6: Multi-GPU Learner-Worker Architecture
**Goal**: Shared policy trained on rigid + cloth simultaneously across GPUs.

**Architecture: Centralized Learner with Async Workers**

```
GPU 0 (Learner + Rigid Actor):
  Actor loop:
    - Collect rigid rollout (16k envs × 32 horizon)
    - Put in learner's inbox

  Learner loop:
    - Take rigid rollout from own actor
    - Check inbox for cloth data from workers (non-blocking)
      - Nothing? Train on rigid only
      - Data arrived? Concatenate, train on combined
    - PPO update (standard single-GPU, no all_reduce)
    - Broadcast updated weights to all workers

GPU 1..N (Cloth Workers):
  loop:
    - Receive latest weights from learner
    - Collect cloth rollout
    - Send rollout buffer to learner (non-blocking)
```

**Key properties:**
- No deadlocks (no collective ops, only point-to-point send/recv)
- No speed bottleneck (learner never waits, always has rigid data)
- Self-adjusting ratio: if cloth is 3x slower, ~25% of training data is cloth
- Scales to N workers trivially
- Staleness handled by PPO clipping (add V-trace if needed)

**Implementation:**
- Modify rl_games training loop to support learner-worker mode
- Workers: strip training code, just collect + send + receive
- Learner: accept variable-size combined buffer, standard PPO
- Communication: `torch.distributed.isend` / `irecv` for non-blocking transfer
- Weight broadcast: learner sends `model.state_dict()` after each update

**What to figure out:**
- Data transfer format (full rollout buffer? just obs+actions+rewards+dones?)
- Transfer size and latency (likely a few hundred MB, < 1 second on NVLink/PCIe)
- Whether PPO clipping alone handles staleness or need V-trace
- How to handle different observation shapes if rigid and cloth have different obs
  (likely same shape since both use point clouds, but verify)
- How to configure torchrun to launch heterogeneous processes

---

## Dependency Graph

```
M0 (proof of life) ← START HERE
 └─ M1 (observations)
     └─ M2 (cloth trajectories)
         └─ M3 (rewards & terminations)
             └─ M4 (curriculum & DR)
                 └─ M5 (validation)
                     └─ M6 (multi-GPU learner-worker)
```

M0-M5 are single-GPU, 100% cloth envs.
M6 adds multi-GPU to combine with rigid training.

## Config Structure

```yaml
# configs/base.yaml additions
cloth:
  enabled: false              # master switch

# configs/layers/tasks/cloth.yaml
cloth:
  enabled: true
  size: [0.3, 0.3, 0.005]    # width, height, thickness
  num_keypoints: 32
  material:
    youngs_modulus: 1e4
    poissons_ratio: 0.45
    dynamic_friction: 0.5
  tasks:
    slide: true
    flatten: true
    fold: true

simulation:
  replicate_physics: false    # required for deformable
  num_envs: 4096              # fewer due to deformable cost

trajectory:
  segments:                   # cloth-specific segments
    - name: "approach"
      type: "waypoint"
      duration: 1.5
    - name: "pinch"
      type: "waypoint"
      duration: 1.0
    - name: "manipulate"
      type: "waypoint"
      duration: 3.0
    - name: "release"
      type: "waypoint"
      duration: 1.5
      hand_coupling: "none"
```

## Open Questions

- How many FEM nodes does a thin cuboid produce? (determines if 32 keypoints is feasible
  or if we need to subsample from more)
- What Young's modulus range gives realistic cloth behavior with thin FEM?
- What's the FPS impact of deformable? (determines feasible num_envs for cloth)
- Can we create crumpled/wrinkled initial states efficiently?
  Options: procedural Z noise, pre-simulated library, kinematic node driving
- Same policy network for rigid + cloth? (likely yes — both produce point cloud obs
  of same shape, both output joint actions)
- Does PPO clipping handle the staleness in learner-worker setup, or do we need V-trace?
