import torch
import torch.nn as nn

from y2r.models.blocks import EfficientUpdateFormer
from y2r.models.embeddings import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed_from_grid,
)


class AutoregressiveIntentTracker(nn.Module):
    """
    Intent tracker that rolls trajectories autoregressively.
    Each iteration builds the next-timestep token block for all query points at once,
    enabling iterative refinement while still leveraging batched attention.
    """

    def __init__(
        self,
        num_future_steps=10,
        hidden_size=384,
        model_resolution=(224, 224),
        add_space_attn=True,
        vit_model_name='dinov2_vits14',
        vit_frozen=False,
        time_depth=6,
        space_depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        p_drop_attn=0.0,
        frame_stack=1,
        cache_quantized_position_encoding=False,
    ):
        super().__init__()
        self.num_future_steps = num_future_steps
        self.hidden_size = hidden_size
        self.model_resolution = model_resolution
        self.frame_stack = frame_stack
        self.cache_quantized_position_encoding = cache_quantized_position_encoding

        # ViT encoder (provides all visual context)
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_model_name)
        self.vit.requires_grad_(not vit_frozen)

        # Token dimensions
        self.query_dim = hidden_size
        self.temporal_dim = hidden_size

        # Temporal embeddings for future predictions
        time_grid = torch.linspace(0, num_future_steps - 1, num_future_steps).reshape(
            1, num_future_steps, 1
        )
        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.temporal_dim, time_grid[0])
        )

        # Observation temporal embeddings for past frames
        obs_time_grid = torch.linspace(-(frame_stack - 1), 0, frame_stack).reshape(
            1, frame_stack, 1
        )
        self.register_buffer(
            "obs_time_emb", get_1d_sincos_pos_embed_from_grid(hidden_size, obs_time_grid[0])
        )

        # Transformer that produces Δx, Δy increments
        self.updateformer = EfficientUpdateFormer(
            depth=time_depth,
            input_dim=hidden_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=2,
            mlp_ratio=mlp_ratio,
            add_space_attn=add_space_attn,
            p_drop_attn=p_drop_attn,
            linear_layer_for_vis_conf=False,
        )

    def extract_vit_features(self, frame):
        """
        Extract DINOv2 patch features from frames with temporal encoding.

        Args:
            frame: (B, T_obs, 3, 224, 224) - RGB images where T_obs = frame_stack
                   (ImageNet normalized in dataloader)

        Returns:
            scene_tokens: (B, T_obs*num_patches, feature_dim) - ViT patch embeddings
                         with temporal encoding, concatenated across frames
        """
        B, T_obs, C, H, W = frame.shape
        assert T_obs == self.frame_stack, f"Expected {self.frame_stack} frames, got {T_obs}"

        # Flatten temporal dimension: (B, T_obs, 3, H, W) -> (B*T_obs, 3, H, W)
        frame_flat = frame.view(B * T_obs, C, H, W)

        # Process all frames through ViT
        vit_output = self.vit.forward_features(frame_flat)
        scene_tokens = vit_output['x_norm_patchtokens']  # (B*T_obs, num_patches, feature_dim)

        # Get dimensions
        num_patches = scene_tokens.shape[1]
        feature_dim = scene_tokens.shape[2]

        # Reshape to separate batch and temporal dimensions
        scene_tokens = scene_tokens.view(B, T_obs, num_patches, feature_dim)

        # Add temporal encoding to distinguish frames
        obs_time_encoding = self.obs_time_emb.unsqueeze(2)  # (1, T_obs, 1, feature_dim)
        scene_tokens = scene_tokens + obs_time_encoding

        # Concatenate tokens from all frames
        scene_tokens = scene_tokens.view(B, T_obs * num_patches, feature_dim)
        return scene_tokens

    def _encode_coords(self, coords):
        """
        Convert normalized coordinates into 2D sinusoidal embeddings.
        Args:
            coords: (B, N, 2) - normalized [0, 1] coordinates (x, y)
        Returns:
            (B, N, hidden_size) embeddings.
        """
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError(f"coords must have shape (B, N, 2), got {coords.shape}")

        height, width = self.model_resolution
        scale = coords.new_tensor([width, height])
        coords_pixel = coords * scale  # convert to pixel grid
        grid = coords_pixel.permute(2, 0, 1)  # (2, B, N)
        coord_embed = get_2d_sincos_pos_embed_from_grid(self.query_dim, grid)
        coord_embed = coord_embed.squeeze(0).reshape(coords.shape[0], coords.shape[1], self.query_dim)
        return coord_embed

    def _build_step_token(self, coords, step_idx):
        """
        Create the transformer token for a single timestep.
        Args:
            coords: (B, N, 2) - coordinates used for this step's context.
            step_idx: int - timestep index (0-based).
        Returns:
            token tensor of shape (B, N, 1, hidden_size).
        """
        if step_idx >= self.time_emb.shape[1]:
            raise ValueError(
                f"step_idx {step_idx} exceeds available embeddings ({self.time_emb.shape[1]}). "
                "Increase num_future_steps during initialization."
            )

        coord_embed = self._encode_coords(coords)  # (B, N, H)
        time_embed = self.time_emb[:, step_idx, :].view(1, 1, 1, self.temporal_dim)
        token = coord_embed.unsqueeze(2) + time_embed  # (B, N, 1, H)
        return token

    def forward(
        self,
        frames,
        query_coords,
        teacher_forcing_disp=None,
        teacher_forcing_mask=None,
        scene_tokens=None,
        num_steps=None,
        return_intermediate=False,
    ):
        """
        Autoregressive rollout over num_steps timesteps.

        Args:
            frames: (B, frame_stack, 3, 224, 224) - RGB frames (normalized).
            query_coords: (B, N, 2) - initial positions in normalized coordinates.
            teacher_forcing_disp: optional (B, N, T, 2) tensor of cumulative GT displacements.
                                 When provided, the model will use these positions to build
                                 tokens (teacher forcing) before predicting the displacement increment.
            teacher_forcing_mask: optional bool tensor (B, N, T) indicating where to apply
                                  teacher forcing. Defaults to all True if teacher data is provided.
            scene_tokens: optional pre-computed ViT tokens. If None, frames must be provided.
            num_steps: override number of autoregressive steps (<= num_future_steps).
            return_intermediate: if True, also return predicted coordinates per step.

        Returns:
            pred_disp: (B, N, num_steps, 2) cumulative displacement predictions.
            optional coords: (B, N, num_steps, 2) predicted coordinates when return_intermediate=True.
        """
        if scene_tokens is None:
            if frames is None:
                raise ValueError("Either frames or scene_tokens must be provided.")
            scene_tokens = self.extract_vit_features(frames)

        if num_steps is None:
            num_steps = self.num_future_steps
        if num_steps > self.num_future_steps:
            raise ValueError(
                f"Requested {num_steps} steps, but model initialized with "
                f"{self.num_future_steps} embeddings."
            )

        B, N, _ = query_coords.shape
        device = query_coords.device
        dtype = query_coords.dtype

        pred_disp = torch.zeros(B, N, num_steps, 2, device=device, dtype=dtype)
        cumulative_disp = torch.zeros(B, N, 2, device=device, dtype=dtype)
        current_coords = query_coords

        if teacher_forcing_disp is not None:
            if teacher_forcing_disp.shape[0] != B or teacher_forcing_disp.shape[1] != N:
                raise ValueError(
                    "teacher_forcing_disp must match batch and query dimensions of query_coords."
                )
            if teacher_forcing_disp.shape[2] < num_steps:
                raise ValueError(
                    f"teacher_forcing_disp only has {teacher_forcing_disp.shape[2]} steps, "
                    f"but num_steps={num_steps} requested."
                )

            teacher_prefix = torch.zeros_like(teacher_forcing_disp)
            teacher_prefix[:, :, 1:, :] = teacher_forcing_disp[:, :, :-1, :]
            teacher_coords = query_coords.unsqueeze(2) + teacher_prefix
            teacher_coords = teacher_coords[:, :, :num_steps, :]

            if teacher_forcing_mask is None:
                teacher_forcing_mask = torch.ones(
                    B, N, num_steps, device=teacher_coords.device, dtype=torch.bool
                )
            else:
                if teacher_forcing_mask.shape != (B, N, teacher_forcing_disp.shape[2]):
                    raise ValueError(
                        "teacher_forcing_mask must have shape (B, N, T) where T "
                        "matches teacher_forcing_disp timesteps."
                    )
                teacher_forcing_mask = teacher_forcing_mask[:, :, :num_steps]

            teacher_forcing_mask = teacher_forcing_mask.unsqueeze(-1)  # (B, N, steps, 1)
        else:
            teacher_coords = None
            teacher_forcing_mask = None

        coord_history = [] if return_intermediate else None

        for step in range(num_steps):
            if teacher_coords is not None:
                tf_coord = teacher_coords[:, :, step, :]
                mask = teacher_forcing_mask[:, :, step, :]
                coords_for_token = torch.where(mask, tf_coord, current_coords)
            else:
                coords_for_token = current_coords

            step_token = self._build_step_token(coords_for_token, step)
            delta = self.updateformer(step_token, scene_tokens, add_space_attn=True)
            delta = delta.squeeze(2)  # (B, N, 2)

            cumulative_disp = cumulative_disp + delta
            pred_disp[:, :, step, :] = cumulative_disp
            current_coords = query_coords + cumulative_disp

            if return_intermediate:
                coord_history.append(current_coords)

        if return_intermediate:
            coords = torch.stack(coord_history, dim=2)  # (B, N, steps, 2)
            return pred_disp, coords
        return pred_disp

    def compute_loss(self, batch):
        """
        Compute supervised loss using teacher forcing for stability.
        """
        frames = batch['frames']
        query_coords = batch['query_coords']
        gt_disp_normalized = batch['gt_disp_normalized']
        disp_std = batch['disp_std']

        pred_disp = self(
            frames=frames,
            query_coords=query_coords,
            teacher_forcing_disp=gt_disp_normalized,
        )

        device = pred_disp.device
        std_tensor = torch.tensor(disp_std, device=device, dtype=pred_disp.dtype)

        pred_disp_denorm = pred_disp * std_tensor
        gt_disp_denorm = gt_disp_normalized[:, :, :pred_disp.shape[2], :] * std_tensor

        error = torch.norm(pred_disp_denorm - gt_disp_denorm, dim=-1)
        loss_mask = torch.ones_like(error)
        loss_mask[:, :, 0] = 0.0

        masked_error = error * loss_mask
        loss = masked_error.sum() / loss_mask.sum().clamp_min(1.0)
        return loss

    @torch.no_grad()
    def predict(self, frames, query_coords, num_steps=None, return_intermediate=False):
        """
        Inference-time rollout without teacher forcing.
        """
        return self(
            frames=frames,
            query_coords=query_coords,
            teacher_forcing_disp=None,
            scene_tokens=None,
            num_steps=num_steps,
            return_intermediate=return_intermediate,
        )

