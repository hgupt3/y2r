"""
Autoregressive Intent Tracker model.

This module implements an autoregressive model for trajectory prediction.
The model predicts trajectories step-by-step, using previous predictions
as context for future predictions.

Training: Uses parallel computation with causal masking (T times faster!)
Inference: Sequential generation (one timestep at a time)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple

from y2r.models.base_model import BaseIntentTracker
from y2r.models.model_config import ENCODING_DIMS


class AutoregressiveIntentTracker(BaseIntentTracker):
    """
    Autoregressive Intent Tracker.
    
    This model predicts future trajectory displacements step-by-step,
    using previous predictions as state context for subsequent predictions.
    
    Architecture:
    - ViT encoder extracts scene tokens from observation frames
    - Track tokens include position + state (cumulative displacement) + temporal
    - UpdateFormer processes tokens with cross-attention
    - Linear heads predict per-step displacement increments
    
    Training Efficiency:
    - Uses PARALLEL computation with teacher forcing + causal mask
    - All T timesteps processed in ONE transformer pass
    - ~T times faster than sequential training!
    
    Inference:
    - Sequential generation (must predict t before t+1)
    - Can use KV caching for further speedup (future work)
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
        # Track Token Encoder (Autoreg: includes state encoding)
        # =========================================================================
        # Track state projection: cumulative displacement -> track_state dim
        self.track_state_proj = nn.Linear(self.coord_dim, self.enc_dims['track_state'])
        
        # Track token: concat(position, state, temporal) -> MLP -> hidden_size
        track_input_dim = (self.enc_dims['track_position'] + 
                         self.enc_dims['track_state'] + 
                         self.enc_dims['temporal'])
        self.track_encoder = self._create_track_encoder(track_input_dim)
        
        # =========================================================================
        # Hand Token Encoder (Optional, with state)
        # =========================================================================
        if hand_mode is not None:
            # Hand rotation projection: 6D -> hand_rotation dim
            self.hand_rot_proj = nn.Linear(6, self.enc_dims['hand_rotation'])
            # Hand state projection: 9D (UVD + rotation displacement) -> hand_state dim
            self.hand_state_proj = nn.Linear(9, self.enc_dims['hand_state'])
            
            # Hand: concat(position, rotation, state, temporal) -> MLP -> hidden_size
            hand_input_dim = (self.enc_dims['hand_position'] + 
                            self.enc_dims['hand_rotation'] + 
                            self.enc_dims['hand_state'] +
                            self.enc_dims['temporal'])
            self.hand_encoder = self._create_hand_encoder(hand_input_dim)
            
            # Hand output head: hidden_size -> 9 (predicts 3 UVD + 6 rotation increment)
            self.hand_head = nn.Linear(self.hidden_size, 9)
        
        # =========================================================================
        # Transformer and Output Heads
        # =========================================================================
        self.updateformer = self._create_updateformer()
        
        # Track output head: hidden_size -> coord_dim (predicts increment)
        self.track_head = nn.Linear(self.hidden_size, self.coord_dim)

    # =========================================================================
    # PARALLEL Token Building (for training with teacher forcing)
    # =========================================================================
    
    def _build_all_track_tokens(
        self, 
        query_coords: torch.Tensor, 
        all_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build track tokens for ALL timesteps at once (parallel training).
        
        Args:
            query_coords: (B, N, coord_dim) - Query positions in [0, 1]
            all_states: (B, N, T, coord_dim) - Cumulative displacements at each timestep
                        From teacher forcing: all_states[:,:,t,:] = gt_disp[:,:,t,:]
            
        Returns:
            track_tokens: (B, N, T, hidden_size)
        """
        B, N, T, _ = all_states.shape
        
        # Encode query positions -> (B, N, pos_dim)
        pos_emb = self._encode_position(query_coords, coord_type='track')
        # Expand for all timesteps: (B, N, T, pos_dim)
        pos_emb = pos_emb.unsqueeze(2).expand(B, N, T, -1)
        
        # Encode states for all timesteps -> (B, N, T, state_dim)
        state_emb = self.track_state_proj(all_states)
        
        # Get temporal embeddings for all timesteps: (T, temporal_dim) -> (1, 1, T, temporal_dim)
        time_emb = self.time_emb[:T].unsqueeze(0).unsqueeze(0)
        time_emb = time_emb.expand(B, N, T, -1)
        
        # Concatenate: (B, N, T, pos_dim + state_dim + temporal_dim)
        token_input = torch.cat([pos_emb, state_emb, time_emb], dim=-1)
        
        # Encode through MLP: (B, N, T, hidden_size)
        track_tokens = self.track_encoder(token_input)
        
        return track_tokens
    
    def _build_all_hand_tokens(
        self,
        hand_query_uvd: torch.Tensor,
        hand_query_rot: torch.Tensor,
        all_hand_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build hand tokens for ALL timesteps at once (parallel training).
        
        Args:
            hand_query_uvd: (B, H, 3) - Hand UVD positions
            hand_query_rot: (B, H, 6) - Hand 6D rotations
            all_hand_states: (B, H, T, 9) - Cumulative hand displacements (UVD + rot)
            
        Returns:
            hand_tokens: (B, H, T, hidden_size)
        """
        B, H, T, _ = all_hand_states.shape
        
        # Encode hand positions -> (B, H, hand_position_dim)
        pos_emb = self._encode_position(hand_query_uvd, coord_type='hand')
        pos_emb = pos_emb.unsqueeze(2).expand(B, H, T, -1)
        
        # Project rotation -> (B, H, hand_rotation_dim)
        rot_emb = self.hand_rot_proj(hand_query_rot)
        rot_emb = rot_emb.unsqueeze(2).expand(B, H, T, -1)
        
        # Encode states -> (B, H, T, hand_state_dim)
        state_emb = self.hand_state_proj(all_hand_states)
        
        # Get temporal embeddings
        time_emb = self.time_emb[:T].unsqueeze(0).unsqueeze(0)
        time_emb = time_emb.expand(B, H, T, -1)
        
        # Concatenate: (B, H, T, pos_dim + rot_dim + state_dim + temporal_dim)
        token_input = torch.cat([pos_emb, rot_emb, state_emb, time_emb], dim=-1)
        
        # Encode through MLP: (B, H, T, hidden_size)
        hand_tokens = self.hand_encoder(token_input)
        
        return hand_tokens

    # =========================================================================
    # SEQUENTIAL Token Building (for inference)
    # =========================================================================
    
    def _build_track_token(
        self, 
        query_coords: torch.Tensor, 
        cumulative_disp: torch.Tensor, 
        timestep: int
    ) -> torch.Tensor:
        """
        Build track token for a single timestep (sequential inference).
        
        Args:
            query_coords: (B, N, coord_dim) - Query positions in [0, 1]
            cumulative_disp: (B, N, coord_dim) - Cumulative displacement so far
            timestep: Current timestep index
            
        Returns:
            track_token: (B, N, 1, hidden_size)
        """
        B, N, _ = query_coords.shape
        
        # Encode query positions -> (B, N, pos_dim)
        pos_emb = self._encode_position(query_coords, coord_type='track')
        
        # Encode state (cumulative displacement) -> (B, N, state_dim)
        state_emb = self.track_state_proj(cumulative_disp)
        
        # Get temporal embedding for this timestep: (temporal_dim,)
        time_emb = self.time_emb[timestep]  # (temporal_dim,)
        time_emb = time_emb.unsqueeze(0).unsqueeze(0).expand(B, N, -1)  # (B, N, temporal_dim)
        
        # Concatenate: (B, N, pos_dim + state_dim + temporal_dim)
        token_input = torch.cat([pos_emb, state_emb, time_emb], dim=-1)
        
        # Encode through MLP: (B, N, hidden_size)
        track_token = self.track_encoder(token_input)
        
        # Add sequence dimension: (B, N, 1, hidden_size)
        return track_token.unsqueeze(2)

    def _build_hand_token(
        self,
        hand_query_uvd: torch.Tensor,
        hand_query_rot: torch.Tensor,
        cumulative_hand_disp: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """
        Build hand token for a single timestep (sequential inference).
        """
        B, H, _ = hand_query_uvd.shape
        
        pos_emb = self._encode_position(hand_query_uvd, coord_type='hand')
        rot_emb = self.hand_rot_proj(hand_query_rot)
        state_emb = self.hand_state_proj(cumulative_hand_disp)
        time_emb = self.time_emb[timestep].unsqueeze(0).unsqueeze(0).expand(B, H, -1)
        
        token_input = torch.cat([pos_emb, rot_emb, state_emb, time_emb], dim=-1)
        hand_token = self.hand_encoder(token_input)
        
        return hand_token.unsqueeze(2)

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        query_coords: Optional[torch.Tensor] = None,
        teacher_forcing_disp: Optional[torch.Tensor] = None,
        scene_tokens: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        return_intermediate: bool = False,
        hand_query_uvd: Optional[torch.Tensor] = None,
        hand_query_rot: Optional[torch.Tensor] = None,
        teacher_forcing_hand_uvd: Optional[torch.Tensor] = None,
        teacher_forcing_hand_rot: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive forward pass.
        
        Training (teacher_forcing_disp provided):
            - PARALLEL: All timesteps processed in one transformer pass with causal mask
            - ~T times faster than sequential!
            
        Inference (no teacher_forcing_disp):
            - SEQUENTIAL: One timestep at a time
        """
        T = num_steps if num_steps is not None else self.num_future_steps
        
        # Get scene tokens
        if scene_tokens is None:
            assert frames is not None, "Either frames or scene_tokens must be provided"
            scene_tokens = self.extract_vit_features(frames, depth)
        
        # Add text tokens if text_mode is enabled
        if self.text_mode and text is not None:
            text_tokens = self.encode_text(text)
            scene_tokens = torch.cat([text_tokens, scene_tokens], dim=1)
        
        B, N, _ = query_coords.shape
        device = query_coords.device
        
        # Check for hand mode
        num_hands = 0
        if self.hand_mode is not None and hand_query_uvd is not None:
            num_hands = hand_query_uvd.shape[1]
        
        # =====================================================================
        # PARALLEL PATH: Training with teacher forcing
        # =====================================================================
        if teacher_forcing_disp is not None:
            T_tf = teacher_forcing_disp.shape[2]
            T = min(T, T_tf)
            
            # Build state sequence for all timesteps
            # State at t is the cumulative displacement UP TO t (so state at t=0 is zeros)
            # For training, we shift: state[t] = gt_disp[t-1] cumulative
            # Actually simpler: state[t] = gt_disp[:,:,t,:] (displacement at t)
            # But autoreg uses cumulative, so we need cumsum
            all_states = torch.zeros(B, N, T, self.coord_dim, device=device)
            # State at t is cumulative sum of displacements 0..t-1
            # state[0] = 0 (no previous displacement)
            # state[1] = gt_disp[0]
            # state[2] = gt_disp[0] + gt_disp[1]
            # etc.
            if T > 1:
                cumsum = torch.cumsum(teacher_forcing_disp[:, :, :T-1, :], dim=2)  # (B, N, T-1, coord_dim)
                all_states[:, :, 1:, :] = cumsum
            
            # Build all track tokens: (B, N, T, hidden_size)
            track_tokens = self._build_all_track_tokens(query_coords, all_states)
            
            # Handle hands
            hand_tokens = None
            if num_hands > 0 and teacher_forcing_hand_uvd is not None:
                # Build hand state sequence similarly
                all_hand_states = torch.zeros(B, num_hands, T, 9, device=device)
                if T > 1:
                    hand_uvd_cumsum = torch.cumsum(teacher_forcing_hand_uvd[:, :, :T-1, :], dim=2)
                    all_hand_states[:, :, 1:, :3] = hand_uvd_cumsum
                    if teacher_forcing_hand_rot is not None:
                        hand_rot_cumsum = torch.cumsum(teacher_forcing_hand_rot[:, :, :T-1, :], dim=2)
                        all_hand_states[:, :, 1:, 3:] = hand_rot_cumsum
                
                hand_tokens = self._build_all_hand_tokens(
                    hand_query_uvd, hand_query_rot, all_hand_states
                )
            
            # Combine track and hand tokens
            if hand_tokens is not None:
                all_tokens = torch.cat([track_tokens, hand_tokens], dim=1)  # (B, N+H, T, hidden_size)
            else:
                all_tokens = track_tokens
            
            # Process through transformer with CAUSAL MASK
            transformer_output = self.updateformer(
                all_tokens, scene_tokens, causal_time=True
            )  # (B, N+H, T, hidden_size)
            
            # Extract outputs and apply heads
            track_output = transformer_output[:, :N, :, :]  # (B, N, T, hidden_size)
            track_disp = self.track_head(track_output)  # (B, N, T, coord_dim)
            
            outputs = {'track_disp': track_disp}
            
            if num_hands > 0 and hand_tokens is not None:
                hand_output = transformer_output[:, N:, :, :]  # (B, H, T, hidden_size)
                hand_pred = self.hand_head(hand_output)  # (B, H, T, 9)
                outputs['hand_uvd_disp'] = hand_pred[..., :3]
                outputs['hand_rot_disp'] = hand_pred[..., 3:]
            
            return outputs
        
        # =====================================================================
        # SEQUENTIAL PATH: Inference without teacher forcing
        # =====================================================================
        else:
            cumulative_disp = torch.zeros(B, N, self.coord_dim, device=device)
            cumulative_hand_disp = None
            if num_hands > 0:
                cumulative_hand_disp = torch.zeros(B, num_hands, 9, device=device)
            
            all_track_disp = []
            all_hand_uvd_disp = []
            all_hand_rot_disp = []
            intermediate = [] if return_intermediate else None
            
            for t in range(T):
                # Build tokens for this timestep only
                track_token = self._build_track_token(query_coords, cumulative_disp, t)
                
                hand_token = None
                if num_hands > 0:
                    hand_token = self._build_hand_token(
                        hand_query_uvd, hand_query_rot, cumulative_hand_disp, t
                    )
                
                # Combine tokens
                if hand_token is not None:
                    all_tokens = torch.cat([track_token, hand_token], dim=1)
                else:
                    all_tokens = track_token
                
                # Process through transformer (no causal mask needed - only 1 timestep)
                transformer_output = self.updateformer(all_tokens, scene_tokens)
                
                # Extract track output and predict increment
                track_output = transformer_output[:, :N, 0, :]
                track_increment = self.track_head(track_output)
                all_track_disp.append(track_increment)
                
                # Update cumulative state for next iteration
                cumulative_disp = cumulative_disp + track_increment
                
                # Handle hands
                if num_hands > 0:
                    hand_output = transformer_output[:, N:, 0, :]
                    hand_pred = self.hand_head(hand_output)
                    all_hand_uvd_disp.append(hand_pred[..., :3])
                    all_hand_rot_disp.append(hand_pred[..., 3:])
                    cumulative_hand_disp = cumulative_hand_disp + hand_pred
                
                if return_intermediate:
                    step_result = {'track_disp': torch.stack(all_track_disp, dim=2)}
                    if num_hands > 0:
                        step_result['hand_uvd_disp'] = torch.stack(all_hand_uvd_disp, dim=2)
                        step_result['hand_rot_disp'] = torch.stack(all_hand_rot_disp, dim=2)
                    intermediate.append(step_result)
            
            outputs = {'track_disp': torch.stack(all_track_disp, dim=2)}
            if num_hands > 0:
                outputs['hand_uvd_disp'] = torch.stack(all_hand_uvd_disp, dim=2)
                outputs['hand_rot_disp'] = torch.stack(all_hand_rot_disp, dim=2)
            
            if return_intermediate:
                return outputs, intermediate
            return outputs

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with teacher forcing (uses PARALLEL path).
        """
        outputs = self(
            frames=batch['frames'],
            query_coords=batch['query_coords'],
            teacher_forcing_disp=batch['gt_disp_normalized'],
            depth=batch.get('depth'),
            hand_query_uvd=batch.get('hand_query_uvd'),
            hand_query_rot=batch.get('hand_query_rot'),
            teacher_forcing_hand_uvd=batch.get('gt_hand_uvd_disp'),
            teacher_forcing_hand_rot=batch.get('gt_hand_rot_disp'),
            text=batch.get('text'),
        )
        
        pred_track_disp = outputs['track_disp']
        gt_track_disp = batch['gt_disp_normalized']
        
        T_pred = pred_track_disp.shape[2]
        gt_track_disp = gt_track_disp[:, :, :T_pred, :]
        
        # Track loss with t=0 masked
        track_error = torch.norm(pred_track_disp - gt_track_disp, dim=-1)
        track_mask = torch.ones_like(track_error)
        track_mask[:, :, 0] = 0.0
        masked_track_error = track_error * track_mask
        track_loss = masked_track_error.sum() / track_mask.sum().clamp_min(1.0)
        
        total_loss = track_loss
        hand_uvd_loss = torch.tensor(0.0, device=track_loss.device)
        hand_rot_loss = torch.tensor(0.0, device=track_loss.device)
        
        # Hand losses
        if self.hand_mode is not None and 'hand_uvd_disp' in outputs:
            pred_hand_uvd = outputs['hand_uvd_disp']
            pred_hand_rot = outputs['hand_rot_disp']
            gt_hand_uvd = batch.get('gt_hand_uvd_disp')
            gt_hand_rot = batch.get('gt_hand_rot_disp')
            
            if gt_hand_uvd is not None:
                gt_hand_uvd = gt_hand_uvd[:, :, :pred_hand_uvd.shape[2], :]
                hand_uvd_error = torch.norm(pred_hand_uvd - gt_hand_uvd, dim=-1)
                hand_uvd_mask = torch.ones_like(hand_uvd_error)
                hand_uvd_mask[:, :, 0] = 0.0
                hand_uvd_loss = (hand_uvd_error * hand_uvd_mask).sum() / hand_uvd_mask.sum().clamp_min(1.0)
                
                if gt_hand_rot is not None:
                    gt_hand_rot = gt_hand_rot[:, :, :pred_hand_rot.shape[2], :]
                    hand_rot_error = torch.norm(pred_hand_rot - gt_hand_rot, dim=-1)
                    hand_rot_mask = torch.ones_like(hand_rot_error)
                    hand_rot_mask[:, :, 0] = 0.0
                    hand_rot_loss = (hand_rot_error * hand_rot_mask).sum() / hand_rot_mask.sum().clamp_min(1.0)
                
                total_loss = track_loss + hand_uvd_loss + 0.5 * hand_rot_loss
        
        return {
            'total_loss': total_loss,
            'track_loss': track_loss,
            'hand_uvd_loss': hand_uvd_loss,
            'hand_rot_loss': hand_rot_loss,
        }

    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor,
        query_coords: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        return_intermediate: bool = False,
        hand_query_uvd: Optional[torch.Tensor] = None,
        hand_query_rot: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-time rollout (uses SEQUENTIAL path).
        """
        return self(
            frames=frames,
            query_coords=query_coords,
            teacher_forcing_disp=None,  # No teacher forcing → sequential path
            scene_tokens=None,
            depth=depth,
            num_steps=num_steps,
            return_intermediate=return_intermediate,
            hand_query_uvd=hand_query_uvd,
            hand_query_rot=hand_query_rot,
            teacher_forcing_hand_uvd=None,
            teacher_forcing_hand_rot=None,
            text=text,
        )
