"""
Direct prediction Intent Tracker model.

This module implements a direct (non-autoregressive, non-diffusion) model
for trajectory prediction. Given observation frames and query coordinates,
it directly predicts all future displacements in a single forward pass.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict

from y2r.models.base_model import BaseIntentTracker
from y2r.models.model_config import ENCODING_DIMS


class IntentTracker(BaseIntentTracker):
    """
    Direct prediction Intent Tracker.
    
    This model predicts all future trajectory displacements in a single forward pass,
    without iterative refinement (autoregressive) or denoising (diffusion).
    
    Architecture:
    - ViT encoder extracts scene tokens from observation frames
    - Track tokens are built from query positions + temporal encodings
    - UpdateFormer processes scene + track tokens with cross-attention
    - Linear heads predict displacement outputs
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
        from_pretrained: bool = True,
        hand_mode: Optional[str] = None,
        text_mode: bool = False,
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
        
        # =========================================================================
        # Track Token Encoder (Direct model: no state encoding needed)
        # =========================================================================
        # Track token: concat(position, temporal) -> MLP -> hidden_size
        track_input_dim = self.enc_dims['track_position'] + self.enc_dims['temporal']
        self.track_encoder = self._create_track_encoder(track_input_dim)
        
        # =========================================================================
        # Hand Token Encoder (Optional)
        # =========================================================================
        if hand_mode is not None:
            # Hand: concat(position, rotation, temporal) -> MLP -> hidden_size
            hand_input_dim = (self.enc_dims['hand_position'] + 
                            self.enc_dims['hand_rotation'] + 
                            self.enc_dims['temporal'])
            self.hand_encoder = self._create_hand_encoder(hand_input_dim)
            
            # Linear projection for 6D rotation -> hand_rotation dim
            self.hand_rot_proj = nn.Linear(6, self.enc_dims['hand_rotation'])
            
            # Hand output head: hidden_size -> 9 (3 UVD + 6 rotation)
            self.hand_head = nn.Linear(self.hidden_size, 9)
        
        # =========================================================================
        # Transformer and Output Heads
        # =========================================================================
        self.updateformer = self._create_updateformer()
        
        # Track output head: hidden_size -> coord_dim (3 for 3D UVD, 2 for 2D)
        self.track_head = nn.Linear(self.hidden_size, self.coord_dim)
    
    def _build_track_tokens(
        self, 
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build track tokens for all query points and timesteps.
        
        Args:
            query_coords: (B, N, coord_dim) - Query positions in [0, 1]
            
        Returns:
            track_tokens: (B, N, T, hidden_size)
        """
        B, N, _ = query_coords.shape
        T = self.num_future_steps
        
        # Encode query positions -> (B, N, pos_dim)
        pos_emb = self._encode_position(query_coords, coord_type='track')
        
        # Expand for all timesteps: (B, N, 1, pos_dim) -> (B, N, T, pos_dim)
        pos_emb = pos_emb.unsqueeze(2).expand(B, N, T, -1)
        
        # Get temporal embeddings: (T, temporal_dim) -> (1, 1, T, temporal_dim)
        time_emb = self.time_emb.unsqueeze(0).unsqueeze(0)
        time_emb = time_emb.expand(B, N, T, -1)
        
        # Concatenate: (B, N, T, pos_dim + temporal_dim)
        token_input = torch.cat([pos_emb, time_emb], dim=-1)
        
        # Encode through MLP: (B, N, T, hidden_size)
        track_tokens = self.track_encoder(token_input)
        
        return track_tokens
    
    def _build_hand_tokens(
        self,
        hand_query_uvd: torch.Tensor,
        hand_query_rot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build hand tokens for all hands and timesteps.
        
        Args:
            hand_query_uvd: (B, H, 3) - Hand UVD positions in [0, 1]
            hand_query_rot: (B, H, 6) - Hand 6D rotations
            
        Returns:
            hand_tokens: (B, H, T, hidden_size)
        """
        B, H, _ = hand_query_uvd.shape
        T = self.num_future_steps
        
        # Encode hand positions -> (B, H, hand_position_dim)
        pos_emb = self._encode_position(hand_query_uvd, coord_type='hand')
        
        # Project rotation -> (B, H, hand_rotation_dim)
        rot_emb = self.hand_rot_proj(hand_query_rot)
        
        # Expand for all timesteps
        pos_emb = pos_emb.unsqueeze(2).expand(B, H, T, -1)  # (B, H, T, pos_dim)
        rot_emb = rot_emb.unsqueeze(2).expand(B, H, T, -1)  # (B, H, T, rot_dim)
        
        # Get temporal embeddings
        time_emb = self.time_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, T, temporal_dim)
        time_emb = time_emb.expand(B, H, T, -1)
        
        # Concatenate: (B, H, T, pos_dim + rot_dim + temporal_dim)
        token_input = torch.cat([pos_emb, rot_emb, time_emb], dim=-1)
        
        # Encode through MLP: (B, H, T, hidden_size)
        hand_tokens = self.hand_encoder(token_input)
        
        return hand_tokens
    
    def forward(
        self,
        frames: torch.Tensor,
        query_coords: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        hand_query_uvd: Optional[torch.Tensor] = None,
        hand_query_rot: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for direct prediction.
        
        Args:
            frames: (B, frame_stack, 3, H, W) - RGB observation frames
            query_coords: (B, N, coord_dim) - Query positions in [0, 1]
            depth: (B, frame_stack, 1, H, W) - Depth maps (3D mode only)
            hand_query_uvd: (B, H, 3) - Hand initial UVD positions
            hand_query_rot: (B, H, 6) - Hand initial 6D rotations
            text: List[str] of length B - Text descriptions
            
        Returns:
            Dict with:
                'track_disp': (B, N, T, coord_dim) - Predicted displacements
                'hand_uvd_disp': (B, H, T, 3) - Hand UVD displacements (if hand_mode)
                'hand_rot_disp': (B, H, T, 6) - Hand rotation displacements (if hand_mode)
        """
        B, N, _ = query_coords.shape
        T = self.num_future_steps
        
        # Extract visual features from observation frames
        scene_tokens = self.extract_vit_features(frames, depth)  # (B, num_scene_tokens, hidden_size)
        
        # Add text tokens if text_mode is enabled
        if self.text_mode and text is not None:
            text_tokens = self.encode_text(text)  # (B, 1, hidden_size)
            scene_tokens = torch.cat([text_tokens, scene_tokens], dim=1)
        
        # Build track tokens: (B, N, T, hidden_size)
        track_tokens = self._build_track_tokens(query_coords)
        
        # Build hand tokens if hand_mode is set
        hand_tokens = None
        num_hands = 0
        if self.hand_mode is not None and hand_query_uvd is not None:
            hand_tokens = self._build_hand_tokens(hand_query_uvd, hand_query_rot)
            num_hands = hand_tokens.shape[1]
        
        # Combine track and hand tokens for transformer
        # Shape: (B, N+H, T, hidden_size)
        if hand_tokens is not None:
            all_tokens = torch.cat([track_tokens, hand_tokens], dim=1)
        else:
            all_tokens = track_tokens
        
        # Process through UpdateFormer
        # Input: (B, N+H, T, hidden_size) query tokens + (B, S, hidden_size) scene tokens
        transformer_output = self.updateformer(all_tokens, scene_tokens)  # (B, N+H, T, hidden_size)
        
        # Split outputs back into track and hand
        track_output = transformer_output[:, :N, :, :]  # (B, N, T, hidden_size)
        
        # Apply track head to get displacements
        track_disp = self.track_head(track_output)  # (B, N, T, coord_dim)
        
        outputs = {'track_disp': track_disp}
        
        # Process hand outputs if available
        if num_hands > 0:
            hand_output = transformer_output[:, N:, :, :]  # (B, H, T, hidden_size)
            hand_pred = self.hand_head(hand_output)  # (B, H, T, 9)
            
            outputs['hand_uvd_disp'] = hand_pred[..., :3]  # (B, H, T, 3)
            outputs['hand_rot_disp'] = hand_pred[..., 3:]  # (B, H, T, 6)
        
        return outputs
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            batch: Dict containing:
                - 'frames': (B, frame_stack, 3, H, W)
                - 'query_coords': (B, N, coord_dim)
                - 'gt_disp_normalized': (B, N, T, coord_dim) - Ground truth displacements
                - 'depth': (B, frame_stack, 1, H, W) - Optional
                - 'hand_query_uvd': (B, H, 3) - Optional
                - 'hand_query_rot': (B, H, 6) - Optional
                - 'gt_hand_uvd_disp': (B, H, T, 3) - Optional
                - 'gt_hand_rot_disp': (B, H, T, 6) - Optional
                - 'text': List[str] - Optional
                
        Returns:
            Dict with 'total_loss', 'track_loss', 'hand_uvd_loss', 'hand_rot_loss'
        """
        # Forward pass
        outputs = self(
            frames=batch['frames'],
            query_coords=batch['query_coords'],
            depth=batch.get('depth'),
            hand_query_uvd=batch.get('hand_query_uvd'),
            hand_query_rot=batch.get('hand_query_rot'),
            text=batch.get('text'),
        )
        
        # Track loss (L1 with t=0 masked out)
        pred_track_disp = outputs['track_disp']
        gt_track_disp = batch['gt_disp_normalized']
        
        track_error = torch.abs(pred_track_disp - gt_track_disp)  # (B, N, T, coord_dim)
        
        # Mask out t=0 (displacement at t=0 should be zero by definition)
        loss_mask = torch.ones_like(track_error[..., 0])  # (B, N, T)
        loss_mask[:, :, 0] = 0.0
        
        masked_track_error = track_error * loss_mask.unsqueeze(-1)
        track_loss = masked_track_error.sum() / (loss_mask.sum() * self.coord_dim)
        
        total_loss = track_loss
        hand_uvd_loss = torch.tensor(0.0, device=track_loss.device)
        hand_rot_loss = torch.tensor(0.0, device=track_loss.device)
        
        # Hand loss (if hand_mode is set)
        if self.hand_mode is not None and 'hand_uvd_disp' in outputs:
            pred_hand_uvd = outputs['hand_uvd_disp']
            pred_hand_rot = outputs['hand_rot_disp']
            gt_hand_uvd = batch.get('gt_hand_uvd_disp')
            gt_hand_rot = batch.get('gt_hand_rot_disp')
            
            hand_uvd_weight = 1.0
            hand_rot_weight = 0.5
            
            if gt_hand_uvd is not None:
                hand_uvd_error = torch.abs(pred_hand_uvd - gt_hand_uvd)
                hand_uvd_mask = torch.ones_like(hand_uvd_error[..., 0])
                hand_uvd_mask[:, :, 0] = 0.0
                masked_hand_uvd_error = hand_uvd_error * hand_uvd_mask.unsqueeze(-1)
                hand_uvd_loss = masked_hand_uvd_error.sum() / (hand_uvd_mask.sum() * 3).clamp_min(1.0)
            
            if gt_hand_rot is not None:
                hand_rot_error = torch.abs(pred_hand_rot - gt_hand_rot)
                hand_rot_mask = torch.ones_like(hand_rot_error[..., 0])
                hand_rot_mask[:, :, 0] = 0.0
                masked_hand_rot_error = hand_rot_error * hand_rot_mask.unsqueeze(-1)
                hand_rot_loss = masked_hand_rot_error.sum() / (hand_rot_mask.sum() * 6).clamp_min(1.0)
            
            total_loss = track_loss + hand_uvd_weight * hand_uvd_loss + hand_rot_weight * hand_rot_loss
        
        return {
            'total_loss': total_loss,
            'track_loss': track_loss,
            'hand_uvd_loss': hand_uvd_loss,
            'hand_rot_loss': hand_rot_loss,
        }
    
    def predict(
        self,
        frames: torch.Tensor,
        query_coords: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        hand_query_uvd: Optional[torch.Tensor] = None,
        hand_query_rot: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-time prediction.
        
        Args:
            frames: (B, frame_stack, 3, H, W) - RGB frames
            query_coords: (B, N, coord_dim) - Initial positions in [0, 1]
            depth: (B, frame_stack, 1, H, W) - Depth maps (3D mode only)
            hand_query_uvd: (B, H, 3) - Hand initial positions
            hand_query_rot: (B, H, 6) - Hand initial rotations
            text: List[str] - Text descriptions
            return_intermediate: bool - Ignored for direct model (API compatibility)
        
        Returns:
            Dict with predictions (same as forward())
            If return_intermediate=True, returns (outputs, []) for API compatibility
        """
        outputs = self(
            frames=frames,
            query_coords=query_coords,
            depth=depth,
            hand_query_uvd=hand_query_uvd,
            hand_query_rot=hand_query_rot,
            text=text,
        )
        
        if return_intermediate:
            return outputs, []
        return outputs
