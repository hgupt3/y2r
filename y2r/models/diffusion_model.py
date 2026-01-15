"""
Diffusion-based Intent Tracker model.

This module implements a diffusion model for trajectory prediction, following the
Diffusion Policy architecture. The model denoises trajectory outputs using DDIM sampling.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from diffusers.schedulers import DDIMScheduler, DDPMScheduler

from y2r.models.base_model import BaseIntentTracker
from y2r.models.embeddings import get_1d_sincos_pos_embed_from_grid
from y2r.models.model_config import ENCODING_DIMS
from y2r.models.blocks import TimestepEmbedder


class DiffusionIntentTracker(BaseIntentTracker):
    """
    Diffusion-based Intent Tracker.
    
    This model uses diffusion/denoising to predict trajectory displacements.
    During training, it learns to predict noise added to ground truth trajectories.
    During inference, it iteratively denoises random noise using DDIM sampling.
    
    Architecture:
    - ViT encoder extracts scene tokens from observation frames
    - Track tokens include position + noisy state + temporal + diffusion timestep
    - UpdateFormer processes tokens with cross-attention
    - Linear heads predict noise to be removed
    """
    
    def __init__(
        self,
        model_size: str = 's',
        num_future_steps: int = 10,
        model_resolution: tuple = (256, 256),
        add_space_attn: bool = True,
        vit_frozen: bool = False,
        p_drop_attn: float = 0.0,
        frame_stack: int = 1,
        track_type: str = '2d',
        num_diffusion_steps: int = 100,
        beta_schedule: str = 'squaredcos_cap_v2',
        num_inference_steps: int = 10,
        disp_mean: Optional[List[float]] = None,
        disp_std: Optional[List[float]] = None,
        from_pretrained: bool = True,
        hand_mode: Optional[str] = None,
        text_mode: bool = False,
        enable_cfg: bool = False,
        cfg_dropout_prob: float = 0.1,
    ):
        # Initialize base class (ViT, text encoder, temporal embeddings)
        super().__init__(
            model_size=model_size,
            num_future_steps=num_future_steps,
            model_resolution=model_resolution,
            add_space_attn=add_space_attn,
            vit_frozen=vit_frozen,
            p_drop_attn=p_drop_attn,
            frame_stack=frame_stack,
            track_type=track_type,
            from_pretrained=from_pretrained,
            hand_mode=hand_mode,
            text_mode=text_mode,
        )

        # CFG configuration
        self.enable_cfg = enable_cfg
        self.cfg_dropout_prob = cfg_dropout_prob

        # Learnable null text embedding for CFG (used when text is dropped)
        # This is a learnable parameter representing "no instruction"
        if text_mode and enable_cfg:
            # Initialize as zeros - will be learned during training like DiT's approach
            # Shape: (1, 1, hidden_size) - single token, will be broadcasted during use
            self.null_text_embedding = nn.Parameter(
                torch.zeros(1, 1, self.hidden_size)
            )
        
        # =========================================================================
        # Diffusion-specific Configuration
        # =========================================================================
        self.total_token_steps = num_future_steps + 1  # Include conditioning slot at t=0
        self.num_diffusion_steps = num_diffusion_steps
        self.default_num_inference_steps = num_inference_steps
        
        # Register displacement normalization stats as buffers
        if disp_mean is not None and disp_std is not None:
            self.register_buffer('disp_mean', torch.tensor(disp_mean, dtype=torch.float32))
            self.register_buffer('disp_std', torch.tensor(disp_std, dtype=torch.float32))
        else:
            self.register_buffer('disp_mean', torch.zeros(self.coord_dim))
            self.register_buffer('disp_std', torch.ones(self.coord_dim))
        
        # Override time embedding to include conditioning slot (T+1 timesteps)
        # Note: squeeze to get (T+1, temporal_dim) instead of (1, T+1, temporal_dim)
        time_grid = torch.linspace(0, self.total_token_steps - 1, self.total_token_steps).reshape(
            1, self.total_token_steps, 1
        )
        time_emb = get_1d_sincos_pos_embed_from_grid(self.enc_dims['temporal'], time_grid[0])
        self.register_buffer("time_emb", time_emb.squeeze(0))  # (T+1, temporal_dim)
        
        # =========================================================================
        # DiT: Diffusion Timestep Embedder
        # =========================================================================
        # Instead of concatenating diffusion timestep into tokens,
        # DiT uses it for adaptive layer normalization conditioning
        self.timestep_embedder = TimestepEmbedder(
            hidden_size=self.hidden_size,
            frequency_embedding_size=256  # Standard DiT embedding size
        )
        
        # =========================================================================
        # Track Token Encoder (DiT: NO diffusion timestep - used for adaLN instead)
        # =========================================================================
        # Track state projection: noisy displacement -> track_state dim
        self.track_state_proj = nn.Linear(self.coord_dim, self.enc_dims['track_state'])

        # Track token: concat(position, state, temporal) -> MLP -> hidden_size
        # Note: diffusion_timestep is NOT concatenated - used for adaLN conditioning
        track_input_dim = (self.enc_dims['track_position'] +
                         self.enc_dims['track_state'] +
                         self.enc_dims['temporal'])
        self.track_encoder = self._create_track_encoder(track_input_dim)
        
        # =========================================================================
        # Hand Token Encoder (DiT: NO diffusion timestep - used for adaLN instead)
        # =========================================================================
        if hand_mode is not None:
            # Hand rotation projection: 6D -> hand_rotation dim
            self.hand_rot_proj = nn.Linear(6, self.enc_dims['hand_rotation'])
            # Hand state projection: 9D (noisy UVD + rotation) -> hand_state dim
            self.hand_state_proj = nn.Linear(9, self.enc_dims['hand_state'])

            # Hand: concat(position, rotation, state, temporal) -> MLP -> hidden_size
            # Note: diffusion_timestep is NOT concatenated - used for adaLN conditioning
            hand_input_dim = (self.enc_dims['hand_position'] +
                            self.enc_dims['hand_rotation'] +
                            self.enc_dims['hand_state'] +
                            self.enc_dims['temporal'])
            self.hand_encoder = self._create_hand_encoder(hand_input_dim)

            # Hand output head: hidden_size -> 9 (predicts noise for 3 UVD + 6 rotation)
            self.hand_head = nn.Linear(self.hidden_size, 9)
        
        # =========================================================================
        # DiT Transformer and Output Heads
        # =========================================================================
        # Use DiT variant with adaptive layer normalization conditioning
        self.updateformer = self._create_updateformer(use_dit=True)
        
        # Track output head: hidden_size -> coord_dim (predicts noise)
        self.track_head = nn.Linear(self.hidden_size, self.coord_dim)
        
        # =========================================================================
        # Diffusion Schedulers
        # =========================================================================
        # DDPM scheduler for training
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            clip_sample=False,
            prediction_type='epsilon',
        )
        
        # DDIM scheduler for fast inference
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            clip_sample=False,
            prediction_type='epsilon',
        )

        # =========================================================================
        # Timestep Importance Sampling (Loss-Aware Sampling)
        # =========================================================================
        # Track per-timestep loss history for importance sampling
        # Based on DiT's LossSecondMomentResampler approach
        self.register_buffer('timestep_loss_history', torch.zeros(num_diffusion_steps, 10))
        self.register_buffer('timestep_loss_counts', torch.zeros(num_diffusion_steps))
        self.timestep_sampling_warmup = 10  # Warmup batches before using importance sampling
        self.timestep_sampling_uniform_prob = 0.1  # Mix 10% uniform sampling
    
    def _build_track_tokens(
        self,
        query_coords: torch.Tensor,
        noisy_traj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build track tokens for all query points and timesteps (DiT version).

        Args:
            query_coords: (B, N, coord_dim) - Query positions in [0, 1]
            noisy_traj: (B, N, T+1, coord_dim) - Noisy trajectory (includes t=0 conditioning)

        Returns:
            track_tokens: (B, N, T+1, hidden_size)

        Note:
            Diffusion timestep is NOT concatenated here - it's used for adaLN conditioning
            in DiTUpdateFormer instead.
        """
        B, N, T_total, _ = noisy_traj.shape

        # Encode query positions -> (B, N, pos_dim)
        pos_emb = self._encode_position(query_coords, coord_type='track')
        # Expand for all timesteps: (B, N, T+1, pos_dim)
        pos_emb = pos_emb.unsqueeze(2).expand(B, N, T_total, -1)

        # Encode noisy state -> (B, N, T+1, state_dim)
        state_emb = self.track_state_proj(noisy_traj)

        # Get temporal embeddings: (T+1, temporal_dim) -> (1, 1, T+1, temporal_dim)
        time_emb = self.time_emb.unsqueeze(0).unsqueeze(0)
        time_emb = time_emb.expand(B, N, T_total, -1)

        # Concatenate embeddings (NO diffusion timestep - used for adaLN)
        token_input = torch.cat([pos_emb, state_emb, time_emb], dim=-1)

        # Encode through MLP: (B, N, T+1, hidden_size)
        track_tokens = self.track_encoder(token_input)

        return track_tokens
    
    def _build_hand_tokens(
        self,
        hand_query_uvd: torch.Tensor,
        hand_query_rot: torch.Tensor,
        noisy_hand_traj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build hand tokens for all hands and timesteps (DiT version).

        Args:
            hand_query_uvd: (B, H, 3) - Hand UVD positions
            hand_query_rot: (B, H, 6) - Hand 6D rotations
            noisy_hand_traj: (B, H, T+1, 9) - Noisy hand trajectory (UVD + rot)

        Returns:
            hand_tokens: (B, H, T+1, hidden_size)

        Note:
            Diffusion timestep is NOT concatenated here - it's used for adaLN conditioning
            in DiTUpdateFormer instead.
        """
        B, H, T_total, _ = noisy_hand_traj.shape

        # Encode hand positions -> (B, H, hand_position_dim)
        pos_emb = self._encode_position(hand_query_uvd, coord_type='hand')
        pos_emb = pos_emb.unsqueeze(2).expand(B, H, T_total, -1)

        # Project rotation -> (B, H, hand_rotation_dim)
        rot_emb = self.hand_rot_proj(hand_query_rot)
        rot_emb = rot_emb.unsqueeze(2).expand(B, H, T_total, -1)

        # Encode noisy state -> (B, H, T+1, hand_state_dim)
        state_emb = self.hand_state_proj(noisy_hand_traj)

        # Get temporal embeddings
        time_emb = self.time_emb.unsqueeze(0).unsqueeze(0)
        time_emb = time_emb.expand(B, H, T_total, -1)

        # Concatenate embeddings (NO diffusion timestep - used for adaLN)
        token_input = torch.cat([pos_emb, rot_emb, state_emb, time_emb], dim=-1)

        # Encode through MLP: (B, H, T+1, hidden_size)
        hand_tokens = self.hand_encoder(token_input)

        return hand_tokens

    # =========================================================================
    # Timestep Importance Sampling Methods
    # =========================================================================

    def update_timestep_loss_history(self, timesteps: torch.Tensor, losses: torch.Tensor):
        """
        Update the per-timestep loss history for importance sampling.

        Args:
            timesteps: (B,) tensor of timestep indices
            losses: (B,) tensor of per-sample losses
        """
        if not self.training:
            return

        # Move to CPU for indexing (buffers are on model device)
        timesteps_cpu = timesteps.cpu()
        losses_cpu = losses.detach().cpu()

        for t, loss in zip(timesteps_cpu, losses_cpu):
            t = t.item()
            # Rolling window: shift and add new loss
            self.timestep_loss_history[t] = torch.roll(self.timestep_loss_history[t], -1)
            self.timestep_loss_history[t, -1] = loss
            self.timestep_loss_counts[t] += 1

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample diffusion timesteps with importance weighting based on loss history.

        During warmup (first few batches), uses uniform sampling.
        After warmup, samples difficult timesteps more frequently based on loss variance.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to place sampled timesteps

        Returns:
            timesteps: (batch_size,) tensor of sampled timestep indices
        """
        # Check if warmed up (all timesteps seen at least once)
        if self.timestep_loss_counts.min() < self.timestep_sampling_warmup:
            # Warmup: uniform random sampling
            return torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device, dtype=torch.long)

        # Compute importance weights from loss history
        # Use second moment (RMS) of losses as in DiT's LossSecondMomentResampler
        loss_means = self.timestep_loss_history.mean(dim=1)  # (num_timesteps,)
        weights = torch.sqrt((self.timestep_loss_history ** 2).mean(dim=1))  # RMS

        # Normalize weights to be a probability distribution
        weights = weights / weights.sum()

        # Mix with uniform sampling (10% uniform, 90% importance)
        uniform_prob = self.timestep_sampling_uniform_prob
        weights = weights * (1 - uniform_prob) + uniform_prob / self.num_diffusion_steps

        # Sample according to weights
        timesteps = torch.multinomial(weights, batch_size, replacement=True).to(device)

        return timesteps

    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        query_coords: Optional[torch.Tensor] = None,
        noisy_traj: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        scene_tokens: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        hand_query_uvd: Optional[torch.Tensor] = None,
        hand_query_rot: Optional[torch.Tensor] = None,
        noisy_hand_traj: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Noise prediction forward pass.
        
        Args:
            frames: (B, frame_stack, C, H, W) - Observation frames
            query_coords: (B, N, coord_dim) - Query positions
            noisy_traj: (B, N, T+1, coord_dim) - Noisy trajectory with t=0 conditioning
            timestep: (B,) - Diffusion timestep
            scene_tokens: Pre-computed scene tokens (optional)
            depth: (B, frame_stack, 1, H, W) - Depth maps (3D mode)
            hand_query_uvd: (B, H, 3) - Hand initial UVD
            hand_query_rot: (B, H, 6) - Hand initial 6D rotation
            noisy_hand_traj: (B, H, T+1, 9) - Noisy hand trajectory
            text: List[str] - Text descriptions
            
        Returns:
            Dict with 'track_noise', optionally 'hand_noise'
        """
        # Get scene tokens
        if scene_tokens is None:
            assert frames is not None, "Either frames or scene_tokens must be provided"
            scene_tokens = self.extract_vit_features(frames, depth)

        # Add text tokens if text_mode is enabled
        if self.text_mode and text is not None:
            text_tokens = self.encode_text(text)

            # CFG: During training, randomly drop text with cfg_dropout_prob
            if self.enable_cfg and self.training:
                B = text_tokens.shape[0]
                # Create dropout mask: True = keep text, False = use null embedding
                keep_text_mask = torch.rand(B, device=text_tokens.device) > self.cfg_dropout_prob

                # Expand null embedding to match batch size and sequence length
                # null_text_embedding: (1, 1, hidden_size) -> (B, L, hidden_size)
                L = text_tokens.shape[1]
                null_tokens = self.null_text_embedding.expand(B, L, -1)

                # Replace text tokens with null where mask is False
                # Shape: keep_text_mask (B,) -> (B, 1, 1) for broadcasting
                keep_text_mask = keep_text_mask.view(B, 1, 1)
                text_tokens = torch.where(keep_text_mask, text_tokens, null_tokens)

            scene_tokens = torch.cat([text_tokens, scene_tokens], dim=1)
        
        B, N, T_total, _ = noisy_traj.shape

        # Process diffusion timestep through embedder for adaLN conditioning
        # timestep: (B,) -> (B, hidden_size)
        timestep_cond = self.timestep_embedder(timestep)

        # Build track tokens: (B, N, T+1, hidden_size)
        track_tokens = self._build_track_tokens(query_coords, noisy_traj)

        # Build hand tokens if available
        hand_tokens = None
        num_hands = 0
        if self.hand_mode is not None and hand_query_uvd is not None and noisy_hand_traj is not None:
            num_hands = hand_query_uvd.shape[1]
            hand_tokens = self._build_hand_tokens(
                hand_query_uvd, hand_query_rot, noisy_hand_traj
            )

        # Combine track and hand tokens
        if hand_tokens is not None:
            all_tokens = torch.cat([track_tokens, hand_tokens], dim=1)
        else:
            all_tokens = track_tokens

        # Process through DiT transformer with timestep conditioning
        transformer_output = self.updateformer(all_tokens, scene_tokens, timestep_cond)
        
        # Extract track output and predict noise
        track_output = transformer_output[:, :N, :, :]  # (B, N, T+1, hidden_size)
        track_noise = self.track_head(track_output)  # (B, N, T+1, coord_dim)
        
        outputs = {'track_noise': track_noise}
        
        # Process hand outputs
        if num_hands > 0:
            hand_output = transformer_output[:, N:, :, :]  # (B, H, T+1, hidden_size)
            hand_noise = self.hand_head(hand_output)  # (B, H, T+1, 9)
            outputs['hand_noise'] = hand_noise
        
        return outputs
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], use_importance_sampling: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.

        Args:
            batch: Dict containing training data
            use_importance_sampling: If True, sample timesteps based on loss history (recommended)

        Returns:
            Dict with 'total_loss', 'track_loss', 'hand_uvd_loss', 'hand_rot_loss',
            'per_sample_loss' (for updating history), and 'timesteps' (for updating history)
        """
        frames = batch['frames']
        query_coords = batch['query_coords']
        gt_disp = batch['gt_disp_normalized']  # (B, N, T, coord_dim)
        depth = batch.get('depth')
        text = batch.get('text')

        B, N, T, _ = gt_disp.shape
        device = gt_disp.device

        # Create conditioning slot (t=0): zero displacement
        zero_slot = torch.zeros(B, N, 1, self.coord_dim, device=device)
        gt_traj = torch.cat([zero_slot, gt_disp], dim=2)  # (B, N, T+1, coord_dim)

        # Sample diffusion timesteps (with optional importance sampling)
        if use_importance_sampling:
            timestep = self.sample_timesteps(B, device)
        else:
            timestep = torch.randint(0, self.num_diffusion_steps, (B,), device=device, dtype=torch.long)
        
        # Sample noise
        noise = torch.randn_like(gt_traj)
        
        # Create conditioning mask (don't noise t=0)
        condition_mask = torch.zeros(B, N, T + 1, self.coord_dim, device=device, dtype=torch.bool)
        condition_mask[:, :, 0, :] = True
        
        # Add noise (respecting conditioning)
        noisy_traj = self.noise_scheduler.add_noise(gt_traj, noise, timestep)
        noisy_traj = torch.where(condition_mask, gt_traj, noisy_traj)
        noise = torch.where(condition_mask, torch.zeros_like(noise), noise)
        
        # Handle hand data
        num_hands = 0
        noisy_hand_traj = None
        hand_noise = None
        gt_hand_traj = None
        hand_condition_mask = None
        
        if self.hand_mode is not None:
            hand_query_uvd = batch.get('hand_query_uvd')
            hand_query_rot = batch.get('hand_query_rot')
            gt_hand_uvd = batch.get('gt_hand_uvd_disp')
            gt_hand_rot = batch.get('gt_hand_rot_disp')
            
            if hand_query_uvd is not None and gt_hand_uvd is not None:
                num_hands = hand_query_uvd.shape[1]
                
                # Combine UVD and rotation displacements
                if gt_hand_rot is not None:
                    gt_hand_disp = torch.cat([gt_hand_uvd, gt_hand_rot], dim=-1)  # (B, H, T, 9)
                else:
                    gt_hand_disp = torch.cat([gt_hand_uvd, torch.zeros(B, num_hands, T, 6, device=device)], dim=-1)
                
                # Create conditioning slot for hands
                hand_zero_slot = torch.zeros(B, num_hands, 1, 9, device=device)
                gt_hand_traj = torch.cat([hand_zero_slot, gt_hand_disp], dim=2)  # (B, H, T+1, 9)
                
                # Sample hand noise
                hand_noise = torch.randn_like(gt_hand_traj)
                
                # Create hand conditioning mask
                hand_condition_mask = torch.zeros(B, num_hands, T + 1, 9, device=device, dtype=torch.bool)
                hand_condition_mask[:, :, 0, :] = True
                
                # Add noise to hand trajectory
                noisy_hand_traj = self.noise_scheduler.add_noise(gt_hand_traj, hand_noise, timestep)
                noisy_hand_traj = torch.where(hand_condition_mask, gt_hand_traj, noisy_hand_traj)
                hand_noise = torch.where(hand_condition_mask, torch.zeros_like(hand_noise), hand_noise)
        
        # Forward pass
        outputs = self(
            frames=frames,
            query_coords=query_coords,
            noisy_traj=noisy_traj,
            timestep=timestep,
            depth=depth,
            hand_query_uvd=batch.get('hand_query_uvd'),
            hand_query_rot=batch.get('hand_query_rot'),
            noisy_hand_traj=noisy_hand_traj,
            text=text,
        )
        
        # Compute track loss (MSE on noise prediction)
        predicted_noise = outputs['track_noise']
        
        # Create loss mask (exclude t=0)
        track_loss_mask = torch.ones(B, N, T + 1, device=device)
        track_loss_mask[:, :, 0] = 0.0
        
        track_mse = (predicted_noise - noise) ** 2  # (B, N, T+1, coord_dim)
        masked_track_mse = track_mse * track_loss_mask.unsqueeze(-1)
        track_loss = masked_track_mse.sum() / (track_loss_mask.sum() * self.coord_dim).clamp_min(1.0)

        # Compute per-sample losses for importance sampling
        # Sum over all dimensions except batch to get per-sample loss
        per_sample_track_loss = masked_track_mse.sum(dim=[1, 2, 3]) / (track_loss_mask[0].sum() * self.coord_dim)

        total_loss = track_loss
        hand_uvd_loss = torch.tensor(0.0, device=device)
        hand_rot_loss = torch.tensor(0.0, device=device)

        # Hand loss
        if num_hands > 0 and 'hand_noise' in outputs:
            predicted_hand_noise = outputs['hand_noise']

            hand_loss_mask = torch.ones(B, num_hands, T + 1, device=device)
            hand_loss_mask[:, :, 0] = 0.0

            hand_uvd_weight = 1.0
            hand_rot_weight = 0.5

            # UVD loss
            hand_uvd_mse = (predicted_hand_noise[..., :3] - hand_noise[..., :3]) ** 2
            masked_uvd_mse = hand_uvd_mse * hand_loss_mask.unsqueeze(-1)
            hand_uvd_loss = masked_uvd_mse.sum() / (hand_loss_mask.sum() * 3).clamp_min(1.0)

            # Per-sample hand UVD loss
            per_sample_hand_uvd = masked_uvd_mse.sum(dim=[1, 2, 3]) / (hand_loss_mask[0].sum() * 3)

            # Rotation loss
            hand_rot_mse = (predicted_hand_noise[..., 3:] - hand_noise[..., 3:]) ** 2
            masked_rot_mse = hand_rot_mse * hand_loss_mask.unsqueeze(-1)
            hand_rot_loss = masked_rot_mse.sum() / (hand_loss_mask.sum() * 6).clamp_min(1.0)

            # Per-sample hand rotation loss
            per_sample_hand_rot = masked_rot_mse.sum(dim=[1, 2, 3]) / (hand_loss_mask[0].sum() * 6)

            # Combine per-sample losses
            per_sample_loss = per_sample_track_loss + hand_uvd_weight * per_sample_hand_uvd + hand_rot_weight * per_sample_hand_rot

            total_loss = track_loss + hand_uvd_weight * hand_uvd_loss + hand_rot_weight * hand_rot_loss
        else:
            per_sample_loss = per_sample_track_loss

        return {
            'total_loss': total_loss,
            'track_loss': track_loss,
            'hand_uvd_loss': hand_uvd_loss,
            'hand_rot_loss': hand_rot_loss,
            'per_sample_loss': per_sample_loss,  # For updating loss history
            'timesteps': timestep,  # For updating loss history
        }
    
    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor,
        query_coords: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        return_intermediate: bool = False,
        hand_query_uvd: Optional[torch.Tensor] = None,
        hand_query_rot: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
        cfg_guidance_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        DDIM sampling for inference.

        Args:
            frames: (B, frame_stack, C, H, W) - Observation frames
            query_coords: (B, N, coord_dim) - Query positions
            depth: (B, frame_stack, 1, H, W) - Depth maps (3D mode)
            num_inference_steps: Number of DDIM steps (default: self.default_num_inference_steps)
            return_intermediate: Whether to return intermediate denoising steps
            hand_query_uvd: (B, H, 3) - Hand initial UVD
            hand_query_rot: (B, H, 6) - Hand initial 6D rotation
            text: List[str] - Text descriptions
            cfg_guidance_scale: Guidance scale for CFG (1.0 = no guidance, >1.0 = stronger conditioning)
                              Only used if enable_cfg=True and text_mode=True

        Returns:
            Dict with 'track_disp', optionally 'hand_uvd_disp', 'hand_rot_disp'
            If return_intermediate, returns (outputs, intermediate_list)
        """
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps
        
        B, N, _ = query_coords.shape
        T = self.num_future_steps
        device = query_coords.device
        
        # Pre-compute scene tokens
        vit_features = self.extract_vit_features(frames, depth)

        # Determine if we need CFG
        use_cfg = self.enable_cfg and self.text_mode and text is not None and cfg_guidance_scale != 1.0

        # Prepare conditional and unconditional scene tokens
        if self.text_mode and text is not None:
            text_tokens = self.encode_text(text)
            cond_scene_tokens = torch.cat([text_tokens, vit_features], dim=1)

            # Prepare unconditional scene tokens for CFG
            if use_cfg:
                # Use null embedding instead of text
                L = text_tokens.shape[1]
                null_tokens = self.null_text_embedding.expand(B, L, -1)
                uncond_scene_tokens = torch.cat([null_tokens, vit_features], dim=1)
            else:
                uncond_scene_tokens = None

            scene_tokens = cond_scene_tokens
        else:
            scene_tokens = vit_features
            cond_scene_tokens = vit_features
            uncond_scene_tokens = None
            use_cfg = False
        
        # Initialize noisy trajectory from random noise
        noisy_traj = torch.randn(B, N, T + 1, self.coord_dim, device=device)
        
        # Create conditioning data (t=0 is zero displacement)
        condition_data = torch.zeros(B, N, T + 1, self.coord_dim, device=device)
        condition_mask = torch.zeros(B, N, T + 1, self.coord_dim, device=device, dtype=torch.bool)
        condition_mask[:, :, 0, :] = True
        
        # Handle hand data
        num_hands = 0
        noisy_hand_traj = None
        hand_condition_data = None
        hand_condition_mask = None
        
        if self.hand_mode is not None and hand_query_uvd is not None:
            num_hands = hand_query_uvd.shape[1]
            noisy_hand_traj = torch.randn(B, num_hands, T + 1, 9, device=device)
            hand_condition_data = torch.zeros(B, num_hands, T + 1, 9, device=device)
            hand_condition_mask = torch.zeros(B, num_hands, T + 1, 9, device=device, dtype=torch.bool)
            hand_condition_mask[:, :, 0, :] = True
        
        # Set inference scheduler timesteps
        self.inference_scheduler.set_timesteps(num_inference_steps)
        
        intermediate = [] if return_intermediate else None
        
        # DDIM sampling loop
        for t in self.inference_scheduler.timesteps:
            timestep = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Apply conditioning
            noisy_traj = torch.where(condition_mask, condition_data, noisy_traj)
            if noisy_hand_traj is not None:
                noisy_hand_traj = torch.where(hand_condition_mask, hand_condition_data, noisy_hand_traj)
            
            # Forward pass to predict noise (with CFG if enabled)
            if use_cfg:
                # Batched CFG: combine conditional and unconditional in single forward pass
                # This is 2x faster than separate passes (DiT-style optimization)

                # Double batch dimension by concatenating conditional and unconditional
                batched_query_coords = torch.cat([query_coords, query_coords], dim=0)
                batched_noisy_traj = torch.cat([noisy_traj, noisy_traj], dim=0)
                batched_timestep = torch.cat([timestep, timestep], dim=0)
                batched_scene_tokens = torch.cat([cond_scene_tokens, uncond_scene_tokens], dim=0)

                # Handle hand data if present
                batched_hand_query_uvd = None
                batched_hand_query_rot = None
                batched_noisy_hand_traj = None
                if hand_query_uvd is not None:
                    batched_hand_query_uvd = torch.cat([hand_query_uvd, hand_query_uvd], dim=0)
                    batched_hand_query_rot = torch.cat([hand_query_rot, hand_query_rot], dim=0)
                if noisy_hand_traj is not None:
                    batched_noisy_hand_traj = torch.cat([noisy_hand_traj, noisy_hand_traj], dim=0)

                # Single forward pass with doubled batch
                batched_outputs = self(
                    frames=None,
                    query_coords=batched_query_coords,
                    noisy_traj=batched_noisy_traj,
                    timestep=batched_timestep,
                    scene_tokens=batched_scene_tokens,
                    hand_query_uvd=batched_hand_query_uvd,
                    hand_query_rot=batched_hand_query_rot,
                    noisy_hand_traj=batched_noisy_hand_traj,
                )

                # Split results: first half is conditional, second half is unconditional
                batched_track_noise = batched_outputs['track_noise']
                cond_track_noise, uncond_track_noise = batched_track_noise.chunk(2, dim=0)

                # Apply CFG: noise_pred = uncond + guidance_scale * (cond - uncond)
                predicted_track_noise = (
                    uncond_track_noise +
                    cfg_guidance_scale * (cond_track_noise - uncond_track_noise)
                )

                # Blend hand noise predictions if available
                if num_hands > 0 and 'hand_noise' in batched_outputs:
                    batched_hand_noise = batched_outputs['hand_noise']
                    cond_hand_noise, uncond_hand_noise = batched_hand_noise.chunk(2, dim=0)
                    predicted_hand_noise = (
                        uncond_hand_noise +
                        cfg_guidance_scale * (cond_hand_noise - uncond_hand_noise)
                    )
                else:
                    predicted_hand_noise = None
            else:
                # Standard prediction without CFG
                outputs = self(
                    frames=None,
                    query_coords=query_coords,
                    noisy_traj=noisy_traj,
                    timestep=timestep,
                    scene_tokens=scene_tokens,
                    hand_query_uvd=hand_query_uvd,
                    hand_query_rot=hand_query_rot,
                    noisy_hand_traj=noisy_hand_traj,
                )
                predicted_track_noise = outputs['track_noise']
                predicted_hand_noise = outputs.get('hand_noise')

            # Denoise track trajectory
            noisy_traj = self.inference_scheduler.step(predicted_track_noise, t, noisy_traj).prev_sample
            noisy_traj = torch.where(condition_mask, condition_data, noisy_traj)

            # Denoise hand trajectory
            if noisy_hand_traj is not None and predicted_hand_noise is not None:
                noisy_hand_traj = self.inference_scheduler.step(predicted_hand_noise, t, noisy_hand_traj).prev_sample
                noisy_hand_traj = torch.where(hand_condition_mask, hand_condition_data, noisy_hand_traj)
            
            # Store intermediate
            if return_intermediate:
                step_result = {'track_disp': noisy_traj[:, :, 1:, :].clone()}
                if noisy_hand_traj is not None:
                    step_result['hand_uvd_disp'] = noisy_hand_traj[:, :, 1:, :3].clone()
                    step_result['hand_rot_disp'] = noisy_hand_traj[:, :, 1:, 3:].clone()
                intermediate.append(step_result)
        
        # Extract final predictions (skip conditioning slot)
        clean_disp = noisy_traj[:, :, 1:, :]  # (B, N, T, coord_dim)
        result = {'track_disp': clean_disp}
        
        if num_hands > 0 and noisy_hand_traj is not None:
            clean_hand_traj = noisy_hand_traj[:, :, 1:, :]  # (B, H, T, 9)
            result['hand_uvd_disp'] = clean_hand_traj[..., :3]
            result['hand_rot_disp'] = clean_hand_traj[..., 3:]
        
        if return_intermediate:
            return result, intermediate
        return result
