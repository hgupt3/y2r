# Repository Guidelines

## Project Structure & Module Organization
This repo combines trajectory prediction (“IntentTracker”) with Isaac Lab RL policies.

Key locations:
- `train.py`: main IntentTracker training entrypoint.
- `configs/`: training and dataset YAML configs (for example, `configs/train_diffusion.yaml`).
- `y2r/`: core Python package.
- `y2r/models/`: model variants (direct, diffusion, autoregressive).
- `y2r/dataloaders/`: dataset loading and augmentations.
- `dataset_scripts/`: multi-stage video-to-dataset pipeline.
- `isaac_scripts/`: shell wrappers for Isaac Lab training/eval/distillation.
- `real_world_execution/`: ROS2 deployment package.
- `IsaacLab/`: Git submodule; treat as a separately versioned project.
- `thirdparty/`: vendored external dependencies; avoid modifying unless required.

## Build, Test, and Development Commands
Use the correct conda environment for the task.

IntentTracker and dataset scripts (`sam` env):
- `conda activate sam`
- `python train.py --config configs/train_direct.yaml`: train direct model.
- `python train.py --config configs/train_diffusion.yaml`: train diffusion model.
- `python dataset_scripts/preprocess.py`: start dataset pipeline.

Isaac Lab RL (`y2r` env):
- `conda activate y2r`
- `./isaac_scripts/train.sh --continue`: train policy.
- `./isaac_scripts/play.sh --continue`: evaluate policy.

## Coding Style & Naming Conventions
Python conventions:
- Use 4-space indentation and keep functions small and explicit.
- Prefer descriptive `snake_case` for functions, variables, and files.
- Use `PascalCase` for classes.
- Follow existing patterns in `y2r/models/` and `y2r/dataloaders/`.
- Keep config filenames descriptive, such as `train_autoreg.yaml`.

## Testing Guidelines
There is no single root test suite. Choose the closest relevant check:
- IntentTracker: run a small training/inference sanity check via `train.py` with a known-good config.
- Isaac Lab: run `./IsaacLab/isaaclab.sh --test` (executes `pytest` under `IsaacLab/tools`).
- When adding logic, prefer lightweight, focused tests near the changed code.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects, often submodule-scoped:
- Example: `Update IsaacLab submodule (reward improvements)`.

Guidelines:
- Keep the subject line under ~72 characters and state the intent.
- Group changes by concern (for example, IntentTracker vs Isaac Lab).
- PRs should include: what changed, why, how to run, and any config/env requirements.
- If behavior or visuals change, include a brief result summary or screenshots.

## Submodules, Third-Party Code, and Artifacts
- For `IsaacLab/`, commit inside the submodule first, then update the pointer in the root repo.
- Avoid large generated artifacts in the repo. Prefer referencing external storage.
- Archive planning or one-off validation docs/scripts to `.claude/archive/` when complete.
