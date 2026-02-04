# Repository Guidelines

## Project Overview
End-to-end dexterous manipulation system with two main models:

[RGB-D + Text Prompt] -> IntentTracker -> [Predicted Trajectory] -> RL Policy -> [Motor Commands]

IntentTracker: Given a scene and language instruction, predicts future object/hand trajectories (the "what to do"). Trained on human demonstration videos.
RL Policy: Given a trajectory to follow, outputs joint actions for Kuka arm + Allegro hand (the "how to do it"). Trained in Isaac Lab simulation.

Currently trained independently; will be connected for end-to-end execution.

## Repository Structure
- `train.py`: main IntentTracker training entrypoint.
- `configs/`: training and dataset YAML configs (for example, `configs/train_diffusion.yaml`).
- `y2r/`: core IntentTracker Python package.
- `y2r/models/`: model variants (direct, diffusion, autoregressive).
- `y2r/dataloaders/`: dataset loading and augmentations.
- `y2r/visualization.py`: prediction visualization.
- `dataset_scripts/`: multi-stage video-to-dataset pipeline.
- `isaac_scripts/`: shell wrappers for Isaac Lab training/eval/distillation.
- `real_world_execution/`: ROS2 deployment package.
- `IsaacLab/`: Git submodule; treat as a separately versioned project.
- `thirdparty/`: vendored external dependencies (SAM2, CoTracker, WiLoR, etc.); avoid modifying unless required.

## Conda Environments
This project uses two separate conda environments:

1) `sam` - IntentTracker and dataset scripts
- `conda activate sam`
- Train: `python train.py --config configs/train_diffusion.yaml`
- Dataset pipeline: `python dataset_scripts/preprocess.py`

2) `y2r` - Isaac Lab simulation
- `conda activate y2r`
- Train: `./isaac_scripts/train.sh --continue`
- Evaluate: `./isaac_scripts/play.sh --continue`

Always use the correct environment for the task.

## IntentTracker Training
Train models:
- `python train.py --config configs/train_direct.yaml`
- `python train.py --config configs/train_diffusion.yaml`
- `python train.py --config configs/train_autoreg.yaml`

Resume training:
- `python train.py --config configs/train_direct.yaml --checkpoint path/to/ckpt.pt`

Config hierarchy: `configs/train_*.yaml` references `configs/dataset_config.yaml`. Model parameters (num_future_steps, frame_stack) are derived from dataset config.

Model variants:
- Direct: `y2r/models/model.py`
- Diffusion: `y2r/models/diffusion_model.py`
- Autoregressive: `y2r/models/autoreg_model.py`

All models use DINOv2 vision encoder with optional SigLIP text conditioning.

## Dataset Processing Pipeline
Scripts in `dataset_scripts/` process raw videos into training data. Configuration in `dataset_scripts/config.yaml`.

Pipeline stages (run in order):
- `python dataset_scripts/preprocess.py` (extract frames)
- `python dataset_scripts/process_gsam.py` (segment objects with SAM2/SAM3)
- `python dataset_scripts/process_cotracker.py` (track 2D points)
- `python dataset_scripts/process_tapip3d.py` (lift to 3D tracks)
- `python dataset_scripts/process_wilor.py` (extract hand poses)
- `python dataset_scripts/create_h5_dataset.py` (package into HDF5)

## Isaac Lab Simulation
All scripts in `isaac_scripts/` run from repo root:
- Teacher training: `./isaac_scripts/train.sh --continue`
- Evaluate policy: `./isaac_scripts/play.sh --continue` (teacher) and `./isaac_scripts/play.sh --student --continue` (student)
- Student distillation: `./isaac_scripts/distill.sh --t_continue --continue`

Environment variables `Y2R_MODE` and `Y2R_TASK` control configuration loading.

Simulation config location: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/`
- `configs/base.yaml`: default values (single source of truth)
- `configs/layers/`: mode overrides (student.yaml, play.yaml)
- `configs/layers/tasks/`: task-specific (cup.yaml, push.yaml)
- `config_loader.py:get_config(mode, task)`: composes layers into typed `Y2RConfig`

Core components:
- `trajectory_manager.py`: generates object + hand trajectories at reset
- `trajectory_env_cfg.py`: Isaac Lab env config (scene, rewards, terminations)
- `mdp/`: observation terms, rewards, actions

## Real-World Execution (ROS2)
ROS2 package in `real_world_execution/` (requires ROS2 Humble).

- `source real_world_execution/activate_ros.sh`
- `ros2 launch real_world_execution predictor.launch.py`

Key nodes:
- `perception_node.py`: camera input, object segmentation
- `predictor_node.py`: IntentTracker inference
- `hand_estimation_node.py`: WiLoR hand pose estimation
- `visualization_node.py`: debug visualization

## Coding Style & Naming Conventions
Python conventions:
- Use 4-space indentation and keep functions small and explicit.
- Prefer descriptive `snake_case` for functions, variables, and files.
- Use `PascalCase` for classes.
- Follow existing patterns in `y2r/models/` and `y2r/dataloaders/`.
- Keep config filenames descriptive, such as `train_autoreg.yaml`.
- Avoid "safe fallback" logic that masks errors; prefer explicit failures when required data is missing.

## Testing Guidelines
There is no single root test suite. Choose the closest relevant check:
- IntentTracker: run a small training/inference sanity check via `train.py` with a known-good config.
- Isaac Lab: run `./IsaacLab/isaaclab.sh --test` (executes `pytest` under `IsaacLab/tools`).
- When adding logic, prefer lightweight, focused tests near the changed code.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects, often submodule-scoped:
- Example: `Update IsaacLab submodule (reward improvements)`

Guidelines:
- Keep the subject line under ~72 characters and state the intent.
- Group changes by concern (for example, IntentTracker vs Isaac Lab).
- PRs should include: what changed, why, how to run, and any config/env requirements.
- If behavior or visuals change, include a brief result summary or screenshots.

## Submodules, Third-Party Code, and Artifacts
- For `IsaacLab/`, commit inside the submodule first, then update the pointer in the root repo.
- Avoid large generated artifacts in the repo. Prefer referencing external storage.
- Archive planning or one-off validation docs/scripts to `.claude/archive/` when complete.
