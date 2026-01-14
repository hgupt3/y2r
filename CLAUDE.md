# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end dexterous manipulation system with two main models:

```
[RGB-D + Text Prompt] → IntentTracker → [Predicted Trajectory] → RL Policy → [Motor Commands]
```

**IntentTracker**: Given a scene and language instruction, predicts future object/hand trajectories - the "what to do". Trained on human demonstration videos.

**RL Policy**: Given a trajectory to follow, outputs joint actions for Kuka arm + Allegro hand - the "how to do it". Trained in Isaac Lab simulation.

Currently trained independently; will be connected for end-to-end execution.

## Repository Structure

```
y2r/
├── train.py                 # IntentTracker model training
├── configs/                 # Training configs (train_direct.yaml, train_diffusion.yaml)
├── y2r/                     # IntentTracker Python package
│   ├── models/              # Model architectures (direct, diffusion, autoregressive)
│   ├── dataloaders/         # Dataset loading and augmentation
│   └── visualization.py     # Prediction visualization
├── dataset_scripts/         # Video processing pipeline
├── real_world_execution/    # ROS2 package for deployment
├── isaac_scripts/           # Isaac Lab training scripts
├── IsaacLab/                # Submodule: simulation environment
└── thirdparty/              # External dependencies (SAM2, CoTracker, WiLoR, etc.)
```

## IntentTracker Training

```bash
# Train trajectory prediction model
python train.py --config configs/train_direct.yaml      # Direct prediction
python train.py --config configs/train_diffusion.yaml   # Diffusion-based
python train.py --config configs/train_autoreg.yaml     # Autoregressive

# Resume training
python train.py --config configs/train_direct.yaml --checkpoint path/to/ckpt.pt
```

Config hierarchy: `configs/train_*.yaml` references `configs/dataset_config.yaml` for dataset parameters. Model parameters (num_future_steps, frame_stack) are derived from dataset config.

### Model Variants

| Type | File | Description |
|------|------|-------------|
| Direct | `y2r/models/model.py` | Single-shot trajectory prediction |
| Diffusion | `y2r/models/diffusion_model.py` | DDIM denoising for trajectories |
| Autoregressive | `y2r/models/autoreg_model.py` | Step-by-step prediction |

All models use DINOv2 vision encoder with optional SigLIP text conditioning.

## Dataset Processing Pipeline

Scripts in `dataset_scripts/` process raw videos into training data. Configuration in `dataset_scripts/config.yaml`.

```bash
# Pipeline stages (run in order)
python dataset_scripts/preprocess.py           # Extract frames at target FPS
python dataset_scripts/process_gsam.py         # Segment objects with SAM2/SAM3
python dataset_scripts/process_cotracker.py   # Track 2D points
python dataset_scripts/process_tapip3d.py     # Lift to 3D tracks
python dataset_scripts/process_wilor.py       # Extract hand poses
python dataset_scripts/create_h5_dataset.py   # Package into HDF5
```

## Isaac Lab Simulation

All scripts in `isaac_scripts/` run from repo root:

```bash
# Teacher training (16k envs, headless)
./isaac_scripts/train.sh --continue

# Evaluate policy
./isaac_scripts/play.sh --continue                    # Teacher
./isaac_scripts/play.sh --student --continue          # Student

# Student distillation
./isaac_scripts/distill.sh --t_continue --continue
```

Environment variables `Y2R_MODE` and `Y2R_TASK` control configuration loading.

### Simulation Config

**Location**: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/`

- `configs/base.yaml` - All default values (single source of truth)
- `configs/layers/` - Mode overrides (student.yaml, play.yaml)
- `configs/layers/tasks/` - Task-specific (cup.yaml, push.yaml)

`config_loader.py:get_config(mode, task)` composes layers into typed `Y2RConfig`.

### Core Components

| File | Purpose |
|------|---------|
| `trajectory_manager.py` | Generates object + hand trajectories at reset |
| `trajectory_env_cfg.py` | Isaac Lab env config (scene, rewards, terminations) |
| `mdp/` | Observation terms, rewards, actions |

## Real-World Execution

ROS2 package in `real_world_execution/`. Requires ROS2 Humble.

```bash
source real_world_execution/activate_ros.sh
ros2 launch real_world_execution predictor.launch.py
```

### ROS Nodes

| Node | Purpose |
|------|---------|
| `perception_node.py` | Camera input, object segmentation |
| `predictor_node.py` | IntentTracker inference |
| `hand_estimation_node.py` | WiLoR hand pose estimation |
| `visualization_node.py` | Debug visualization |

## Submodule Workflow

IsaacLab is a Git submodule:
```bash
cd IsaacLab && git add -A && git commit -m "message" && git push
cd .. && git add IsaacLab && git commit -m "Update IsaacLab submodule" && git push
```

## Third-Party Dependencies

Located in `thirdparty/`: SAM2, SAM3, CoTracker3, TAPIP3D, WiLoR, DINOv2, diffuEraser, ViPE
