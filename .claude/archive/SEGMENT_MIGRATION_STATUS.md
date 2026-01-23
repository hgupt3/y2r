# Segment-Based Trajectory System Migration Status

## Completed ✓

### 1. Config System (config_loader.py)
- Added `HandCouplingMode` enum (full/position_only/none)
- Added segment dataclasses:
  - `BaseSegmentConfig` with hand_coupling and hand_orientation fields
  - `WaypointSegmentConfig` for fixed/computed waypoints
  - `HelicalSegmentConfig` for helical motion (nut threading)
  - `RandomWaypointSegmentConfig` for training variation
- Added `_parse_segments()` to parse YAML segment configs
- Removed `PhasesConfig` and `WaypointsConfig` dataclasses

### 2. Trajectory Generation (trajectory_manager.py)
- Added segment storage buffers in `__init__`
- Added `_expand_segments()` to expand random_waypoint → N waypoints at reset
- Implemented `_compute_segment_poses()` to resolve segment target poses
- Implemented `_generate_waypoint_segment()` with symmetric ease-in-out
- Implemented `_generate_helical_segment()` for threading motion
- Rewrote `_generate_full_trajectory()` to compose segments
- Updated `get_phase()` to map segments → phase indices for rewards

### 3. Config Files
- Updated `configs/base.yaml` with default segment-based trajectory

## In Progress 🚧

### Hand Coupling Modes
- **Status**: Partially implemented
- **What's Done**: Config fields, segment storage
- **What's Needed**:
  - Modify `_generate_hand_trajectory()` to apply coupling modes
  - POSITION_ONLY: hand position follows object, orientation frozen at segment boundary
  - NONE: hand decouples entirely, goes to release_pose
  - Store hand pose at segment boundaries for position_only mode

## Not Started ❌

### 1. Phase Boundary Calculation
- **Issue**: `t_manip_end`, `t_settle_end` calculation from segments is incomplete
- **Location**: `reset()` method around line 350
- **Needed**: Calculate per-env boundaries by iterating through expanded segments

### 2. Remove Old Code References
- Many methods still reference `cfg.trajectory.phases` and `cfg.waypoints`
- Need to either remove or add compatibility layer

### 3. Task Config Migration
- `configs/layers/tasks/cup.yaml` - needs segment config
- `configs/layers/tasks/pan.yaml` - needs segment config
- `configs/layers/tasks/push.yaml` - needs segment config
- `configs/layers/tasks/insertion.yaml` - needs helical segment

### 4. Remove Settle Phase
- Still referenced in old code
- Should be removed entirely per requirements

## Critical Issues to Fix Before Testing

1. **Old config references will break** - `cfg.trajectory.phases.grasp` etc no longer exist in base.yaml
2. **Hand trajectory generation** - still uses old phase system, doesn't respect coupling modes
3. **Replanning system** - push_t replanning references old phases

## Recommended Next Steps

1. Add compatibility layer to extract phase durations from segments
2. Test basic trajectory generation (object only, no hand)
3. Implement hand coupling modes
4. Migrate all task configs
5. Remove old phase/waypoint code
6. Full integration testing

## Files Modified

- `config_loader.py` - segment system added, old configs removed
- `trajectory_manager.py` - segment generation added, old code still present
- `configs/base.yaml` - converted to segments

## Files To Modify

- `trajectory_manager.py` - finish hand coupling, remove old code
- `configs/layers/tasks/*.yaml` - migrate all 4 tasks
- Potentially `trajectory_env_cfg.py` if it has phase references
