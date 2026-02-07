# Staged Reset Implementation Plan

## Context

The RL policy always resets from scratch (object on table, robot at home). This means it spends most training time on early phases (approach/grasp) and rarely practices later phases (manipulation/release). **Staged reset** captures mid-episode snapshots at segment transitions into a ring buffer. On reset, some fraction of envs restore from the buffer instead of starting fresh, giving the policy direct practice on all phases.

## Design Overview

```
CAPTURE (every step, for segment-transitioning envs):
  detect segment transitions → subsample with probability → snapshot physics + trajectory state → push to ring buffer

RESTORE (at reset time):
  split reset_ids into fresh_ids / staged_ids
  ├─ fresh_ids: normal reset flow (events → trajectory_manager.reset())
  └─ staged_ids: restore physics via event function → restore trajectory in obs function → fix episode_length_buf
```

**Two interception points:**
1. **`mdp/events.py`** — new `staged_reset_restore()` event (fires LAST) overwrites physics state for staged envs
2. **`mdp/observations.py`** — `target_sequence_obs_b.__call__()` restores trajectory state for staged envs (skips `trajectory_manager.reset()`)

Communication between the two: `env._staged_restore_data` dict set by the event, consumed and cleared by the observation function.

---

## 1. Config Changes

### `configs/base.yaml` — add after `randomization:` section

```yaml
# ==============================================================================
# STAGED RESET
# ==============================================================================
staged_reset:
  enabled: false
  buffer_capacity: 2000         # Ring buffer size (num snapshots)
  staged_fraction: 0.5          # Fraction of resets that restore from buffer (rest are fresh)
  capture_probability: 0.1      # Probability of capturing at each segment transition
  min_buffer_fill: 100          # Minimum snapshots before staged resets begin
```

### `config_loader.py` — add dataclass + field

```python
@dataclass
class StagedResetConfig:
    enabled: bool
    buffer_capacity: int
    staged_fraction: float
    capture_probability: float
    min_buffer_fill: int
```

Add `staged_reset: StagedResetConfig` to `Y2RConfig` (between `randomization` and `visualization`).

---

## 2. New File: `staged_reset_buffer.py`

**Location:** `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/staged_reset_buffer.py`

### Pre-allocated GPU Tensors

The buffer stores `capacity` snapshots with pre-allocated tensors:

**Physics state** (stored relative to env_origins for portability):
- `robot_root_pose` (capacity, 7)
- `robot_root_velocity` (capacity, 6)
- `robot_joint_pos` (capacity, num_joints)
- `robot_joint_vel` (capacity, num_joints)
- `object_root_pose` (capacity, 7)
- `object_root_velocity` (capacity, 6)

**Trajectory state** — mirrors TrajectoryManager buffers:
- `traj_trajectory` (capacity, total_targets, 7)
- `traj_hand_trajectory` (capacity, total_targets, 7)
- `traj_phase_time` (capacity,)
- `traj_current_idx` (capacity,)
- `traj_t_grasp_end/t_manip_end/t_episode_end` (capacity,)
- `traj_goal_poses` (capacity, 7), `traj_start_poses` (capacity, 7)
- Segment-level: `segment_poses`, `segment_boundaries`, `segment_durations`, `num_segments`, `coupling_modes`, all segment boolean flags, helical params, custom pose, hand orientation
- Hand-specific: `grasp_pose`, `release_pose`, `start_palm_pose`, `hand_pose_at_segment_boundary`
- Push-T: `last_replan_idx`, `current_object_poses`, `skip_manipulation`

**Metadata:**
- `elapsed_steps` (capacity,) — for episode_length_buf correction

**Memory estimate** (capacity=2000, total_targets=3000):
- trajectory + hand_trajectory: 2 * 2000 * 3000 * 7 * 4B = **336 MB**
- Everything else: < 10 MB
- Total: ~350 MB (fine for training GPUs)

### Key Methods

```python
class StagedResetBuffer:
    def __init__(self, cfg, trajectory_manager, num_joints, device):
        # Pre-allocate all tensors from trajectory_manager dimensions

    @property
    def is_ready(self) -> bool:
        return self.count >= self.cfg.min_buffer_fill

    def capture(self, env, trajectory_manager, capture_ids):
        """Capture physics + trajectory state for capture_ids into ring buffer.

        Reads physics directly from asset data (NOT scene.get_state()) for efficiency.
        Stores poses relative to env_origins.
        """
        n = len(capture_ids)
        # Compute write indices (ring buffer wrap)
        write_indices = (torch.arange(n, device=self.device) + self.write_idx) % self.capacity
        self.write_idx = (self.write_idx + n) % self.capacity
        self.count = min(self.count + n, self.capacity)

        # Physics: read directly from asset data, subtract env_origins
        robot = env.scene["robot"]
        obj = env.scene["object"]
        origins = env.scene.env_origins[capture_ids]

        pose = robot.data.root_pose_w[capture_ids].clone()
        pose[:, :3] -= origins
        self.robot_root_pose[write_indices] = pose
        # ... similarly for velocity, joints, object

        # Trajectory: use trajectory_manager.get_state(capture_ids)
        traj_state = trajectory_manager.get_state(capture_ids)
        self.traj_trajectory[write_indices] = traj_state["trajectory"]
        # ... all fields

        # Elapsed steps
        dt = env.step_dt  # physics_dt * decimation
        self.elapsed_steps[write_indices] = (trajectory_manager.phase_time[capture_ids] / dt).long()

    def sample(self, n) -> dict:
        """Sample n snapshots (with replacement). Returns dict of tensors."""
        indices = torch.randint(0, self.count, (n,), device=self.device)
        return {
            "robot_root_pose": self.robot_root_pose[indices],
            "robot_root_velocity": self.robot_root_velocity[indices],
            "robot_joint_pos": self.robot_joint_pos[indices],
            "robot_joint_vel": self.robot_joint_vel[indices],
            "object_root_pose": self.object_root_pose[indices],
            "object_root_velocity": self.object_root_velocity[indices],
            "elapsed_steps": self.elapsed_steps[indices],
            "trajectory": {k: tensor[indices] for all traj fields},
        }

    def split_reset_ids(self, reset_ids) -> tuple[Tensor, Tensor]:
        """Split into (fresh_ids, staged_ids). Returns all fresh if buffer not ready."""
        if not self.is_ready:
            return reset_ids, torch.tensor([], device=self.device, dtype=torch.long)
        n = len(reset_ids)
        n_fresh = max(1, int(n * (1.0 - self.cfg.staged_fraction)))
        perm = torch.randperm(n, device=self.device)
        return reset_ids[perm[:n_fresh]], reset_ids[perm[n_fresh:]]
```

---

## 3. TrajectoryManager Changes

**File:** `trajectory_manager.py`

### Add `get_state(env_ids)` and `set_state(state, env_ids)`

```python
def get_state(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Snapshot all trajectory state for given env_ids."""
    return {
        "trajectory": self.trajectory[env_ids].clone(),
        "hand_trajectory": self.hand_trajectory[env_ids].clone(),
        "phase_time": self.phase_time[env_ids].clone(),
        "current_idx": self.current_idx[env_ids].clone(),
        "t_grasp_end": self.t_grasp_end[env_ids].clone(),
        "t_manip_end": self.t_manip_end[env_ids].clone(),
        "t_episode_end": self.t_episode_end[env_ids].clone(),
        "start_poses": self.start_poses[env_ids].clone(),
        "goal_poses": self.goal_poses[env_ids].clone(),
        "env_origins": self.env_origins[env_ids].clone(),
        "segment_poses": self.segment_poses[env_ids].clone(),
        "segment_boundaries": self.segment_boundaries[env_ids].clone(),
        "segment_durations": self.segment_durations[env_ids].clone(),
        "num_segments": self.num_segments[env_ids].clone(),
        "coupling_modes": self.coupling_modes[env_ids].clone(),
        "segment_is_grasp": self.segment_is_grasp[env_ids].clone(),
        "segment_is_release": self.segment_is_release[env_ids].clone(),
        "segment_is_goal": self.segment_is_goal[env_ids].clone(),
        "segment_is_return": self.segment_is_return[env_ids].clone(),
        "segment_is_helical": self.segment_is_helical[env_ids].clone(),
        "segment_helical_axis": self.segment_helical_axis[env_ids].clone(),
        "segment_helical_angular_velocity": self.segment_helical_angular_velocity[env_ids].clone(),
        "segment_helical_translation": self.segment_helical_translation[env_ids].clone(),
        "segment_has_custom_pose": self.segment_has_custom_pose[env_ids].clone(),
        "segment_custom_pose": self.segment_custom_pose[env_ids].clone(),
        "segment_has_hand_orientation": self.segment_has_hand_orientation[env_ids].clone(),
        "segment_hand_orientation": self.segment_hand_orientation[env_ids].clone(),
        "hand_pose_at_segment_boundary": self.hand_pose_at_segment_boundary[env_ids].clone(),
        "grasp_pose": self.grasp_pose[env_ids].clone(),
        "release_pose": self.release_pose[env_ids].clone(),
        "start_palm_pose": self.start_palm_pose[env_ids].clone(),
        "skip_manipulation": self.skip_manipulation[env_ids].clone(),
        "last_replan_idx": self.last_replan_idx[env_ids].clone(),
        "current_object_poses": self.current_object_poses[env_ids].clone(),
        # Grasp sampling (for visualization only, but cheap to include)
        "grasp_surface_point": self.grasp_surface_point[env_ids].clone(),
        "grasp_surface_normal": self.grasp_surface_normal[env_ids].clone(),
    }

def set_state(self, state: dict[str, torch.Tensor], env_ids: torch.Tensor):
    """Restore trajectory state for given env_ids from a snapshot."""
    self.trajectory[env_ids] = state["trajectory"]
    self.hand_trajectory[env_ids] = state["hand_trajectory"]
    self.phase_time[env_ids] = state["phase_time"]
    self.current_idx[env_ids] = state["current_idx"]
    # ... all fields (mirror of get_state)
```

### Add `get_current_segment_index()`

```python
def get_current_segment_index(self) -> torch.Tensor:
    """Return current segment index (0-based) for each env. Vectorized."""
    t_expanded = self.phase_time.unsqueeze(1)  # (N, 1)
    after_start = t_expanded >= self.segment_boundaries  # (N, max_segs+1)
    seg_indices = torch.arange(self.segment_boundaries.shape[1], device=self.device).unsqueeze(0)
    seg_indices_masked = torch.where(after_start, seg_indices, torch.full_like(seg_indices, -1))
    return seg_indices_masked.max(dim=1).values  # (N,)
```

This reuses the same logic already in `get_phase()`.

---

## 4. Event Function: `staged_reset_restore()`

**File:** `mdp/events.py`

This event fires **last** in the reset event list. It overwrites physics state for staged envs.

```python
def staged_reset_restore(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Restore physics state from staged reset buffer for a subset of reset envs."""
    buffer = getattr(env, "_staged_reset_buffer", None)
    if buffer is None or not buffer.is_ready:
        return

    # Split into fresh and staged
    fresh_ids, staged_ids = buffer.split_reset_ids(env_ids)
    if len(staged_ids) == 0:
        return

    # Sample snapshots
    snapshots = buffer.sample(len(staged_ids))

    # Restore robot state (add env_origins of TARGET envs)
    robot = env.scene[robot_cfg.name]
    origins = env.scene.env_origins[staged_ids]

    root_pose = snapshots["robot_root_pose"].clone()
    root_pose[:, :3] += origins
    robot.write_root_pose_to_sim(root_pose, env_ids=staged_ids)
    robot.write_root_velocity_to_sim(snapshots["robot_root_velocity"], env_ids=staged_ids)
    robot.write_joint_state_to_sim(
        snapshots["robot_joint_pos"], snapshots["robot_joint_vel"], env_ids=staged_ids
    )
    robot.set_joint_position_target(snapshots["robot_joint_pos"], env_ids=staged_ids)
    robot.set_joint_velocity_target(snapshots["robot_joint_vel"], env_ids=staged_ids)

    # Restore object state
    obj = env.scene[object_cfg.name]
    obj_pose = snapshots["object_root_pose"].clone()
    obj_pose[:, :3] += origins
    obj.write_root_pose_to_sim(obj_pose, env_ids=staged_ids)
    obj.write_root_velocity_to_sim(snapshots["object_root_velocity"], env_ids=staged_ids)

    # Store for observation function to consume
    env._staged_restore_data = {
        "staged_ids": staged_ids,
        "snapshots": snapshots,
    }
```

### Register in `trajectory_env_cfg.py` `_build_events_cfg()`

Add as the LAST event term in `reset` mode so it overwrites all prior randomization:

```python
events.staged_reset = EventTerm(
    func=mdp.staged_reset_restore,
    mode="reset",
)
```

---

## 5. Observation Function Changes

**File:** `mdp/observations.py` — `target_sequence_obs_b`

### In `__init__()` — initialize buffer

```python
# After trajectory_manager creation:
self._staged_reset_buffer = None
self._prev_segment_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

if y2r_cfg.staged_reset.enabled:
    from ..staged_reset_buffer import StagedResetBuffer
    num_joints = self.ref_asset.num_joints
    self._staged_reset_buffer = StagedResetBuffer(
        cfg=y2r_cfg.staged_reset,
        trajectory_manager=self.trajectory_manager,
        num_joints=num_joints,
        device=env.device,
    )
    env._staged_reset_buffer = self._staged_reset_buffer  # Expose for event function
```

### In `__call__()` — modify reset handling (lines 1476-1508)

```python
if len(reset_ids) > 0:
    # Check for staged restore data (set by staged_reset_restore event)
    staged_data = getattr(env, "_staged_restore_data", None)
    staged_ids = staged_data["staged_ids"] if staged_data else torch.tensor([], device=env.device, dtype=torch.long)

    if len(staged_ids) > 0:
        # Fresh ids = reset_ids that are NOT in staged_ids (vectorized)
        fresh_mask = ~torch.isin(reset_ids, staged_ids)
        fresh_ids = reset_ids[fresh_mask]
    else:
        fresh_ids = reset_ids

    # Fresh resets: normal trajectory_manager.reset()
    if len(fresh_ids) > 0:
        # ... existing reset logic for fresh_ids (lines 1478-1508, using fresh_ids instead of reset_ids)

    # Staged resets: restore trajectory state from snapshot
    if len(staged_ids) > 0:
        snapshots = staged_data["snapshots"]
        self.trajectory_manager.set_state(snapshots["trajectory"], staged_ids)

        # Fix episode_length_buf so termination timing is correct
        env.episode_length_buf[staged_ids] = snapshots["elapsed_steps"]

        # Reset _prev_segment_idx to avoid false capture
        self._prev_segment_idx[staged_ids] = self.trajectory_manager.get_current_segment_index()[staged_ids]

    # Clean up
    env._staged_restore_data = None
```

### In `__call__()` — add capture logic (after `trajectory_manager.step()`, before observations)

```python
# Capture segment transitions for staged reset buffer
if self._staged_reset_buffer is not None:
    curr_seg = self.trajectory_manager.get_current_segment_index()
    transitioned = curr_seg > self._prev_segment_idx

    # Exclude envs that just reset (their segment index was just set)
    if len(reset_ids) > 0:
        transitioned[reset_ids] = False

    if transitioned.any():
        candidate_ids = transitioned.nonzero(as_tuple=True)[0]

        # Subsample with capture_probability
        keep = torch.rand(len(candidate_ids), device=env.device) < self._staged_reset_buffer.cfg.capture_probability
        capture_ids = candidate_ids[keep]

        if len(capture_ids) > 0:
            self._staged_reset_buffer.capture(env, self.trajectory_manager, capture_ids)

    self._prev_segment_idx[:] = curr_seg
```

### In `reset()` — add buffer stats logging

```python
if self._staged_reset_buffer is not None:
    metrics["StagedReset/buffer_fill"] = self._staged_reset_buffer.count
    metrics["StagedReset/buffer_utilization"] = self._staged_reset_buffer.count / self._staged_reset_buffer.capacity
```

---

## 6. Edge Cases

| Case | Handling |
|------|----------|
| **Empty buffer** (early training) | `is_ready` returns False, all resets are fresh |
| **More staged_ids than buffer entries** | `sample()` uses `torch.randint` (with replacement) |
| **Push-T replanning** | Snapshot includes `last_replan_idx` and `current_object_poses`; replanning resumes naturally |
| **Object scale mismatch** | Scale is randomized per-reset but snapshot restores pose, not scale. Minor discrepancy but adds robustness |
| **False capture at reset** | `transitioned[reset_ids] = False` prevents capture on reset boundaries |
| **Table state** | Not touched — event only restores robot + object |
| **Observation history** | CircularBuffers reset normally (same transient as fresh reset) |

---

## Files to Modify

| File | Change |
|------|--------|
| `configs/base.yaml` | Add `staged_reset:` section (~8 lines) |
| `config_loader.py` | Add `StagedResetConfig` dataclass + field on `Y2RConfig` (~12 lines) |
| **`staged_reset_buffer.py`** | **NEW** — Ring buffer class (~200 lines) |
| `trajectory_manager.py` | Add `get_state()`, `set_state()`, `get_current_segment_index()` (~80 lines) |
| `mdp/observations.py` | Buffer init, capture logic, restore logic (~60 lines) |
| `mdp/events.py` | Add `staged_reset_restore()` function (~40 lines) |
| `trajectory_env_cfg.py` | Register event term (~5 lines) |

All files are under: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/`

---

## Verification

1. **Unit test**: Run with `staged_reset.enabled: false` — behavior should be identical to current
2. **Buffer fill**: Train for a few hundred steps with `enabled: true`, check `StagedReset/buffer_fill` in wandb increases
3. **Restore correctness**: In play mode with 4 envs, verify staged-reset envs resume mid-trajectory (visual check: object + hand appear mid-grasp/manipulation)
4. **Episode length**: Verify staged envs terminate at the correct time (not running past trajectory end)
5. **Training signal**: Compare learning curves with staged_fraction=0.0 vs 0.5 — later phases (manipulation/release success) should improve faster
