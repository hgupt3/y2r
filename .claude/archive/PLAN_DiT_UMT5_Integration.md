# Implementation Plan: DiT-Style Diffusion + UMT5 Text Encoder

## Overview
Upgrade the diffusion model to use proper DiT (Diffusion Transformer) conditioning with adaptive layer normalization (adaLN), and replace SigLIP with UMT5 for better instruction understanding.

---

## Phase 1: Add DiT Components to blocks.py

### 1.1 Add DiT Timestep Embedder
**File:** `y2r/models/blocks.py`

**What to add:**
```python
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar diffusion timesteps into vector representations.
    Based on DiT (Diffusion Transformer) architecture.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb  # (B, hidden_size)
```

**Key points:**
- Sinusoidal embedding → MLP with SiLU activation
- Output dimension = hidden_size for direct conditioning
- Replaces the current precomputed sincos buffer approach

### 1.2 Add DiT Self-Attention Block
**File:** `y2r/models/blocks.py`

**What to add:**
```python
def modulate(x, shift, scale):
    """Apply affine modulation: scale * x + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTAttnBlock(nn.Module):
    """
    Self-attention block with adaptive layer norm (adaLN-Zero).
    Timestep conditioning via scale/shift/gate parameters.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attn_class=Attention,
        drop=0.0,
        **block_kwargs
    ):
        super().__init__()
        # LayerNorms without learnable parameters (adaLN will provide them)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Attention
        dim_head = hidden_size // num_heads
        self.attn = attn_class(
            hidden_size, num_heads=num_heads, dim_head=dim_head, qkv_bias=True, **block_kwargs
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=drop,
        )

        # Adaptive modulation: produces 6 parameters (shift, scale, gate) x 2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask=None):
        """
        Args:
            x: (B, N, C) input tokens
            c: (B, C) conditioning vector (timestep embedding)
            mask: optional attention mask
        """
        # Split modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention block with modulation
        attn_bias = None
        if mask is not None:
            # Handle mask (same logic as current AttnBlock)
            num_heads = self.attn.heads
            B, seq_len = x.shape[0], x.shape[1]
            mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(B, num_heads, seq_len, seq_len)
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = torch.where(mask_expanded, torch.zeros_like(mask_expanded, dtype=x.dtype),
                                   torch.full_like(mask_expanded, max_neg_value, dtype=x.dtype))

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            attn_bias=attn_bias
        )

        # MLP block with modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x
```

**Key differences from current AttnBlock:**
- Takes conditioning vector `c` (timestep embedding)
- Applies scale/shift modulation to LayerNorm outputs
- Gates the residual connections

### 1.3 Add DiT Cross-Attention Block
**File:** `y2r/models/blocks.py`

**What to add:**
```python
class DiTCrossAttnBlock(nn.Module):
    """
    Cross-attention block with adaptive layer norm (adaLN-Zero).
    Queries attend to context (scene tokens), conditioned on timestep.
    """
    def __init__(
        self,
        hidden_size,
        context_dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        **block_kwargs
    ):
        super().__init__()
        # LayerNorms without learnable parameters
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)  # Context norm is standard
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Cross-attention
        dim_head = hidden_size // num_heads
        self.cross_attn = Attention(
            hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            dim_head=dim_head,
            qkv_bias=True,
            **block_kwargs
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=drop,
        )

        # Adaptive modulation for query and MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, context, c, mask=None):
        """
        Args:
            x: (B, N, C) query tokens
            context: (B, M, C) key/value tokens (scene features)
            c: (B, C) conditioning vector (timestep embedding)
            mask: optional attention mask
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Cross-attention with modulation
        attn_bias = None
        if mask is not None:
            # Handle context masking
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(-1, self.cross_attn.heads, -1, context.shape[1])
            else:
                mask = mask[:, None, None].expand(-1, self.cross_attn.heads, x.shape[1], -1)
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value

        x = x + gate_msa.unsqueeze(1) * self.cross_attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            context=self.norm_context(context),
            attn_bias=attn_bias
        )

        # MLP with modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x
```

### 1.4 Add DiTUpdateFormer
**File:** `y2r/models/blocks.py`

**What to add:**
```python
class DiTUpdateFormer(nn.Module):
    """
    Transformer with DiT-style conditioning for diffusion trajectory prediction.

    Uses factorized space-time attention with timestep conditioning:
    1. Cross-attention to scene tokens (conditioned)
    2. Spatial self-attention (conditioned)
    3. Temporal self-attention (conditioned)

    Each block is modulated by diffusion timestep via adaLN.
    """
    def __init__(
        self,
        depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        p_drop_attn=0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.add_space_attn = add_space_attn

        # Input projection
        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True) if input_dim != hidden_size else nn.Identity()

        # DiT blocks
        self.scene_cross_attn_blocks = nn.ModuleList([
            DiTCrossAttnBlock(
                hidden_size,
                context_dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=p_drop_attn
            ) for _ in range(depth)
        ])

        if add_space_attn:
            self.space_attn_blocks = nn.ModuleList([
                DiTAttnBlock(
                    hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                    drop=p_drop_attn
                ) for _ in range(depth)
            ])

        self.time_attn_blocks = nn.ModuleList([
            DiTAttnBlock(
                hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_class=Attention,
                drop=p_drop_attn
            ) for _ in range(depth)
        ])

        # Initialize weights with DiT-style zero-init
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize with DiT's adaLN-Zero: zero-out gate parameters."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers (gates start at 0)
        for block in self.scene_cross_attn_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.add_space_attn:
            for block in self.space_attn_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.time_attn_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, all_tokens, scene_tokens, timestep_cond, scene_mask=None, causal_mask=None):
        """
        Args:
            all_tokens: (B, N, T, C) - Track (+ hand) tokens
            scene_tokens: (B, M, C) - Scene context tokens (ViT features + text)
            timestep_cond: (B, C) - Diffusion timestep conditioning vector
            scene_mask: optional mask for scene tokens
            causal_mask: optional causal mask for temporal attention

        Returns:
            output: (B, N, T, C) - Updated tokens
        """
        B, N, T, C = all_tokens.shape

        # Input projection
        x = self.input_transform(all_tokens)

        # Process through DiT blocks
        for i in range(self.depth):
            # 1. Cross-attention to scene (all NT tokens attend to scene)
            x_flat = x.view(B, N * T, C)
            x_flat = self.scene_cross_attn_blocks[i](
                x_flat, scene_tokens, timestep_cond, mask=scene_mask
            )
            x = x_flat.view(B, N, T, C)

            # 2. Spatial attention (tokens at same t attend to each other)
            if self.add_space_attn:
                x_space = x.permute(0, 2, 1, 3).reshape(B * T, N, C)  # (B*T, N, C)
                # Repeat timestep_cond for each timestep
                timestep_cond_space = timestep_cond.unsqueeze(1).repeat(1, T, 1).view(B * T, C)
                x_space = self.space_attn_blocks[i](x_space, timestep_cond_space)
                x = x_space.view(B, T, N, C).permute(0, 2, 1, 3)  # (B, N, T, C)

            # 3. Temporal attention (same position across t attend)
            x_time = x.permute(0, 1, 2, 3).reshape(B * N, T, C)  # (B*N, T, C)
            timestep_cond_time = timestep_cond.unsqueeze(1).repeat(1, N, 1).view(B * N, C)
            x_time = self.time_attn_blocks[i](x_time, timestep_cond_time, mask=causal_mask)
            x = x_time.view(B, N, T, C)

        return x
```

**Key points:**
- Factorized attention preserved (memory efficient)
- Every block receives timestep conditioning
- Zero-initialization for stable training start

---

## Phase 2: Replace SigLIP with UMT5

### 2.1 Update model_config.py
**File:** `y2r/models/model_config.py`

**Changes:**
```python
MODEL_SIZE_CONFIGS = {
    's': {
        'hidden_size': 384,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'time_depth': 6,
        'vit_model_name': 'vit_small_patch16_dinov3',
        # UMT5 for text encoding
        'text_model_name': 'google/umt5-small',
        'text_embed_dim': 512,  # UMT5-small hidden size
    },
    'b': {
        'hidden_size': 768,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'time_depth': 12,
        'vit_model_name': 'vit_base_patch16_dinov3',
        'text_model_name': 'google/umt5-base',
        'text_embed_dim': 768,  # Perfect match!
    },
    'l': {
        'hidden_size': 1024,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'time_depth': 24,
        'vit_model_name': 'vit_large_patch16_dinov3',
        'text_model_name': 'google/umt5-large',
        'text_embed_dim': 1024,  # Perfect match!
    },
}
```

**Dimension matching:**
- Small: 512 → 384 (needs projection)
- Base: 768 → 768 (no projection needed!)
- Large: 1024 → 1024 (no projection needed!)

### 2.2 Update base_model.py
**File:** `y2r/models/base_model.py`

**Replace SigLIP section (lines 129-140) with:**
```python
if text_mode:
    from transformers import AutoTokenizer, UMT5EncoderModel
    text_model_name = cfg['text_model_name']
    text_embed_dim = cfg['text_embed_dim']

    self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    self.text_encoder = UMT5EncoderModel.from_pretrained(text_model_name)

    # Projection layer (only needed if dimensions don't match)
    if text_embed_dim != hidden_size:
        self.text_proj = nn.Linear(text_embed_dim, hidden_size)
    else:
        self.text_proj = nn.Identity()

    # Freeze text encoder (never train UMT5)
    for param in self.text_encoder.parameters():
        param.requires_grad = False
```

**Update encode_text method (lines 191-226):**
```python
def encode_text(self, text_list: List[str]) -> torch.Tensor:
    """
    Encode batch of text strings using UMT5 encoder.

    Args:
        text_list: List of B text strings

    Returns:
        text_tokens: (B, L, hidden_size) - Text embeddings projected to hidden_size
                     where L is sequence length (varies based on text)
    """
    if not self.text_mode:
        raise RuntimeError("encode_text called but text_mode is False")

    # Tokenize text
    inputs = self.text_tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=128,  # Longer for instructions
        return_tensors='pt'
    )

    # Move to same device as model
    device = next(self.text_encoder.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Encode with UMT5 encoder (frozen)
    with torch.no_grad():
        outputs = self.text_encoder(**inputs)

    # Use last_hidden_state (all token embeddings)
    text_emb = outputs.last_hidden_state  # (B, L, text_embed_dim)

    # Project to hidden_size
    text_tokens = self.text_proj(text_emb)  # (B, L, hidden_size)

    return text_tokens
```

**Key changes:**
- UMT5EncoderModel (encoder-only)
- Returns full sequence of tokens (not just pooled CLS)
- Always frozen (no unfreeze option)
- Longer max_length (128 vs 64) for instructions

**Update freeze/unfreeze methods to skip text encoder:**
```python
def freeze_encoders(self):
    """Freeze ViT encoder only (text encoder always frozen)."""
    for param in self.vit.parameters():
        param.requires_grad = False
    print("Encoders frozen (ViT)")

def unfreeze_encoders(self):
    """Unfreeze ViT encoder only (text encoder always frozen)."""
    for param in self.vit.parameters():
        param.requires_grad = True
    print("Encoders unfrozen (ViT)")
```

---

## Phase 3: Add Classifier-Free Guidance (CFG)

### 3.1 Add Null Text Embedding
**File:** `y2r/models/base_model.py`

**Add to __init__ after text encoder setup (after line 142):**
```python
if text_mode:
    from transformers import AutoTokenizer, UMT5EncoderModel
    text_model_name = cfg['text_model_name']
    text_embed_dim = cfg['text_embed_dim']

    self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    self.text_encoder = UMT5EncoderModel.from_pretrained(text_model_name)

    # Projection layer
    if text_embed_dim != hidden_size:
        self.text_proj = nn.Linear(text_embed_dim, hidden_size)
    else:
        self.text_proj = nn.Identity()

    # Freeze text encoder
    for param in self.text_encoder.parameters():
        param.requires_grad = False

    # === NEW: Learnable null embedding for CFG ===
    # Initialize with small random values to be learned during training
    self.null_text_embedding = nn.Parameter(
        torch.randn(1, 1, hidden_size) * 0.02
    )
```

**Key points:**
- Null embedding is learnable (trained alongside the model)
- Shape: (1, 1, hidden_size) → broadcasts to (B, 1, hidden_size)
- Used when text conditioning is dropped during training or for unconditional inference

### 3.2 Update encode_text for CFG Support
**File:** `y2r/models/base_model.py`

**Replace encode_text method:**
```python
def encode_text(
    self,
    text_list: Optional[List[str]] = None,
    use_null_embedding: bool = False
) -> torch.Tensor:
    """
    Encode batch of text strings using UMT5 encoder.
    Supports CFG by optionally returning null embeddings.

    Args:
        text_list: List of B text strings (ignored if use_null_embedding=True)
        use_null_embedding: If True, return null embeddings for CFG

    Returns:
        text_tokens: (B, L, hidden_size) - Text embeddings
                     If use_null_embedding: (B, 1, hidden_size)
    """
    if not self.text_mode:
        raise RuntimeError("encode_text called but text_mode is False")

    # CFG: Return null embedding
    if use_null_embedding:
        # Determine batch size
        batch_size = len(text_list) if text_list is not None else 1
        return self.null_text_embedding.expand(batch_size, -1, -1)

    # Normal text encoding
    if text_list is None:
        raise ValueError("text_list required when use_null_embedding=False")

    # Tokenize
    inputs = self.text_tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    # Move to device
    device = next(self.text_encoder.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Encode with UMT5 (frozen)
    with torch.no_grad():
        outputs = self.text_encoder(**inputs)

    # Project to hidden_size
    text_emb = outputs.last_hidden_state  # (B, L, text_embed_dim)
    text_tokens = self.text_proj(text_emb)  # (B, L, hidden_size)

    return text_tokens
```

### 3.3 Add CFG Dropout During Training
**File:** `y2r/models/diffusion_model.py`

**Update forward method to include CFG dropout:**
```python
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
    cfg_dropout_prob: float = 0.1,  # NEW: CFG dropout probability
) -> Dict[str, torch.Tensor]:
    """
    Noise prediction forward pass with DiT conditioning and CFG support.
    """
    # Get scene tokens
    if scene_tokens is None:
        assert frames is not None
        scene_tokens = self.extract_vit_features(frames, depth)

    B = scene_tokens.shape[0]

    # Add text tokens with CFG dropout
    if self.text_mode and text is not None:
        # During training: randomly drop text conditioning
        if self.training and torch.rand(1).item() < cfg_dropout_prob:
            # Use null embedding
            text_tokens = self.encode_text(text, use_null_embedding=True)
        else:
            # Normal text encoding
            text_tokens = self.encode_text(text, use_null_embedding=False)

        scene_tokens = torch.cat([text_tokens, scene_tokens], dim=1)

    # ... rest of forward pass (unchanged) ...
    # Build tokens, get timestep conditioning, process through transformer, etc.
```

**Key changes:**
- Added `cfg_dropout_prob` parameter (default 0.1 = 10% dropout)
- During training: randomly replace text with null embedding
- Model learns both conditional and unconditional predictions

### 3.4 Add CFG Sampling During Inference
**File:** `y2r/models/diffusion_model.py`

**Update predict method for CFG:**
```python
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
    guidance_scale: float = 1.0,  # NEW: CFG guidance scale
) -> Dict[str, torch.Tensor]:
    """
    DDIM sampling for inference with Classifier-Free Guidance (CFG).

    Args:
        guidance_scale: CFG guidance scale
            - 1.0: No guidance (standard conditional sampling)
            - 1.5-2.5: Moderate guidance (recommended)
            - >3.0: Strong guidance (may cause artifacts)

    Returns:
        Dict with 'track_disp', optionally 'hand_uvd_disp', 'hand_rot_disp'
    """
    if num_inference_steps is None:
        num_inference_steps = self.default_num_inference_steps

    B, N, _ = query_coords.shape
    T = self.num_future_steps
    device = query_coords.device

    # Pre-compute scene tokens
    scene_tokens = self.extract_vit_features(frames, depth)

    # Prepare conditional and unconditional contexts for CFG
    use_cfg = self.text_mode and text is not None and guidance_scale != 1.0

    if use_cfg:
        # Conditional: with text
        text_tokens_cond = self.encode_text(text, use_null_embedding=False)
        scene_tokens_cond = torch.cat([text_tokens_cond, scene_tokens], dim=1)

        # Unconditional: null text
        text_tokens_uncond = self.encode_text(text, use_null_embedding=True)
        scene_tokens_uncond = torch.cat([text_tokens_uncond, scene_tokens], dim=1)
    else:
        # Standard inference (no CFG)
        if self.text_mode and text is not None:
            text_tokens = self.encode_text(text, use_null_embedding=False)
            scene_tokens = torch.cat([text_tokens, scene_tokens], dim=1)
        scene_tokens_cond = scene_tokens
        scene_tokens_uncond = None

    # Initialize noisy trajectory
    noisy_traj = torch.randn(B, N, T + 1, self.coord_dim, device=device)

    # Conditioning data (t=0 is zero displacement)
    condition_data = torch.zeros(B, N, T + 1, self.coord_dim, device=device)
    condition_mask = torch.zeros(B, N, T + 1, self.coord_dim, device=device, dtype=torch.bool)
    condition_mask[:, :, 0, :] = True

    # Handle hand data (similar setup)
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

        if use_cfg:
            # === CFG: Run model TWICE ===

            # Unconditional prediction
            outputs_uncond = self(
                frames=None,
                query_coords=query_coords,
                noisy_traj=noisy_traj,
                timestep=timestep,
                scene_tokens=scene_tokens_uncond,
                hand_query_uvd=hand_query_uvd,
                hand_query_rot=hand_query_rot,
                noisy_hand_traj=noisy_hand_traj,
            )
            noise_pred_uncond = outputs_uncond['track_noise']

            # Conditional prediction
            outputs_cond = self(
                frames=None,
                query_coords=query_coords,
                noisy_traj=noisy_traj,
                timestep=timestep,
                scene_tokens=scene_tokens_cond,
                hand_query_uvd=hand_query_uvd,
                hand_query_rot=hand_query_rot,
                noisy_hand_traj=noisy_hand_traj,
            )
            noise_pred_cond = outputs_cond['track_noise']

            # Combine with guidance
            predicted_track_noise = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # Same for hand if present
            if noisy_hand_traj is not None and 'hand_noise' in outputs_cond:
                hand_noise_uncond = outputs_uncond['hand_noise']
                hand_noise_cond = outputs_cond['hand_noise']
                predicted_hand_noise = hand_noise_uncond + guidance_scale * (
                    hand_noise_cond - hand_noise_uncond
                )
            else:
                predicted_hand_noise = None
        else:
            # Standard prediction (no CFG)
            outputs = self(
                frames=None,
                query_coords=query_coords,
                noisy_traj=noisy_traj,
                timestep=timestep,
                scene_tokens=scene_tokens_cond,
                hand_query_uvd=hand_query_uvd,
                hand_query_rot=hand_query_rot,
                noisy_hand_traj=noisy_hand_traj,
            )
            predicted_track_noise = outputs['track_noise']
            predicted_hand_noise = outputs.get('hand_noise')

        # Denoise track trajectory
        noisy_traj = self.inference_scheduler.step(
            predicted_track_noise, t, noisy_traj
        ).prev_sample
        noisy_traj = torch.where(condition_mask, condition_data, noisy_traj)

        # Denoise hand trajectory
        if noisy_hand_traj is not None and predicted_hand_noise is not None:
            noisy_hand_traj = self.inference_scheduler.step(
                predicted_hand_noise, t, noisy_hand_traj
            ).prev_sample
            noisy_hand_traj = torch.where(hand_condition_mask, hand_condition_data, noisy_hand_traj)

        # Store intermediate if requested
        if return_intermediate:
            step_result = {'track_disp': noisy_traj[:, :, 1:, :].clone()}
            if noisy_hand_traj is not None:
                step_result['hand_uvd_disp'] = noisy_hand_traj[:, :, 1:, :3].clone()
                step_result['hand_rot_disp'] = noisy_hand_traj[:, :, 1:, 3:].clone()
            intermediate.append(step_result)

    # Extract final predictions (skip conditioning slot)
    clean_disp = noisy_traj[:, :, 1:, :]
    result = {'track_disp': clean_disp}

    if num_hands > 0 and noisy_hand_traj is not None:
        clean_hand_traj = noisy_hand_traj[:, :, 1:, :]
        result['hand_uvd_disp'] = clean_hand_traj[..., :3]
        result['hand_rot_disp'] = clean_hand_traj[..., 3:]

    if return_intermediate:
        return result, intermediate
    return result
```

**Key changes:**
- Added `guidance_scale` parameter
- When guidance_scale != 1.0: run model twice (conditional + unconditional)
- Combine predictions: `uncond + guidance_scale * (cond - uncond)`
- Default guidance_scale=1.0 (no CFG overhead unless requested)

### 3.5 Update Training Config
**File:** `configs/train_diffusion.yaml`

**Add CFG parameters (optional):**
```yaml
# Model configuration
model:
  model_size: 'b'
  text_mode: true

# Classifier-Free Guidance (CFG) - OPTIONAL
# Set enable_cfg: true to train with CFG support
# Leave false or omit to disable CFG (no training/inference overhead)
enable_cfg: false  # Default: disabled

# CFG training parameters (only used if enable_cfg: true)
cfg_dropout_prob: 0.1  # 10% text dropout during training

# CFG inference parameters (only used if enable_cfg: true)
cfg_guidance_scale: 1.5  # Guidance scale for validation (1.0 = no guidance)
```

**Create separate config for CFG training:**
**File:** `configs/train_diffusion_cfg.yaml`
```yaml
# Inherit from base diffusion config
base_config: train_diffusion.yaml

# Override CFG settings
enable_cfg: true
cfg_dropout_prob: 0.1
cfg_guidance_scale: 2.0  # Stronger guidance for better instruction following
```

### 3.6 Update Training Loop
**File:** `train.py`

**Pass CFG dropout conditionally:**
```python
# In training loop
enable_cfg = cfg.get('enable_cfg', False)

if enable_cfg:
    # Train with CFG dropout
    outputs = model.compute_loss(
        batch,
        cfg_dropout_prob=cfg.get('cfg_dropout_prob', 0.1)
    )
else:
    # Standard training (no CFG)
    outputs = model.compute_loss(batch, cfg_dropout_prob=0.0)
```

**Use CFG during validation conditionally:**
```python
# In validation loop
enable_cfg = cfg.get('enable_cfg', False)
guidance_scale = cfg.get('cfg_guidance_scale', 1.0) if enable_cfg else 1.0

with torch.no_grad():
    predictions = model.predict(
        frames=batch['frames'],
        query_coords=batch['query_coords'],
        text=batch.get('text'),
        guidance_scale=guidance_scale
    )
```

### 3.7 Update compute_loss Method
**File:** `y2r/models/diffusion_model.py`

**Make CFG dropout configurable in compute_loss:**
```python
def compute_loss(
    self,
    batch: Dict[str, torch.Tensor],
    cfg_dropout_prob: float = 0.0  # Default: no CFG (disabled)
) -> Dict[str, torch.Tensor]:
    """
    Compute diffusion training loss.

    Args:
        batch: Dict containing training data
        cfg_dropout_prob: Probability of dropping text conditioning (0.0 = disabled)

    Returns:
        Dict with 'total_loss', 'track_loss', 'hand_uvd_loss', 'hand_rot_loss'
    """
    frames = batch['frames']
    query_coords = batch['query_coords']
    gt_disp = batch['gt_disp_normalized']
    depth = batch.get('depth')
    text = batch.get('text')

    # ... existing noise setup code ...

    # Forward pass with configurable CFG dropout
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
        cfg_dropout_prob=cfg_dropout_prob  # Pass through
    )

    # ... rest of loss computation ...
```

**Key points:**
- `enable_cfg: false` (default): No CFG training or inference overhead
- `enable_cfg: true`: Enables CFG with configurable dropout and guidance
- Separate config file for CFG training keeps base config clean
- Default `cfg_dropout_prob=0.0` in compute_loss means no overhead unless explicitly enabled

---

## Phase 4: Update DiffusionIntentTracker

### 3.1 Replace Timestep Embedding
**File:** `y2r/models/diffusion_model.py`

**Remove (lines 82-97):**
- `time_emb` buffer registration
- `diffusion_time_emb` buffer registration

**Add to __init__ (after line 80):**
```python
# DiT-style timestep embedder
self.timestep_embedder = TimestepEmbedder(hidden_size)
```

### 3.2 Update Transformer Creation
**File:** `y2r/models/diffusion_model.py`

**Replace `_create_updateformer()` in base_model.py:**
```python
def _create_dit_updateformer(self) -> DiTUpdateFormer:
    """Create DiT-conditioned UpdateFormer for diffusion model."""
    return DiTUpdateFormer(
        depth=self.time_depth,
        input_dim=self.hidden_size,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        output_dim=self.hidden_size,
        mlp_ratio=self.mlp_ratio,
        add_space_attn=self.add_space_attn,
        p_drop_attn=self.p_drop_attn,
    )
```

**In DiffusionIntentTracker.__init__ (line 135):**
```python
# Use DiT-conditioned transformer
self.updateformer = self._create_dit_updateformer()
```

### 3.3 Update Token Building
**File:** `y2r/models/diffusion_model.py`

**Simplify `_build_track_tokens` (remove diffusion_timestep concatenation):**
```python
def _build_track_tokens(
    self,
    query_coords: torch.Tensor,
    noisy_traj: torch.Tensor,
) -> torch.Tensor:
    """
    Build track tokens (without diffusion timestep - that's handled separately).

    Args:
        query_coords: (B, N, coord_dim)
        noisy_traj: (B, N, T+1, coord_dim)

    Returns:
        track_tokens: (B, N, T+1, hidden_size)
    """
    B, N, T_total, _ = noisy_traj.shape

    # Position encoding
    pos_emb = self._encode_position(query_coords, coord_type='track')
    pos_emb = pos_emb.unsqueeze(2).expand(B, N, T_total, -1)

    # State encoding
    state_emb = self.track_state_proj(noisy_traj)

    # Temporal encoding
    time_emb = self.time_emb.unsqueeze(0).unsqueeze(0)
    time_emb = time_emb.expand(B, N, T_total, -1)

    # Concatenate (NO diffusion timestep here)
    token_input = torch.cat([pos_emb, state_emb, time_emb], dim=-1)

    # Encode through MLP
    track_tokens = self.track_encoder(token_input)

    return track_tokens
```

**Similarly update `_build_hand_tokens`** (remove diffusion timestep).

**Update track_encoder input_dim calculation:**
```python
# In __init__, line 106
track_input_dim = (self.enc_dims['track_position'] +
                   self.enc_dims['track_state'] +
                   self.enc_dims['temporal'])
# NO diffusion_timestep here!
```

### 3.4 Update Forward Pass
**File:** `y2r/models/diffusion_model.py`

**Replace forward method (lines 250-326):**
```python
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
    Noise prediction forward pass with DiT conditioning.
    """
    # Get scene tokens
    if scene_tokens is None:
        assert frames is not None
        scene_tokens = self.extract_vit_features(frames, depth)

    # Add text tokens (now returns sequence of tokens, not single CLS)
    if self.text_mode and text is not None:
        text_tokens = self.encode_text(text)  # (B, L, hidden_size)
        scene_tokens = torch.cat([text_tokens, scene_tokens], dim=1)

    B, N, T_total, _ = noisy_traj.shape

    # Build track tokens (WITHOUT diffusion timestep)
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

    # DiT: Get timestep conditioning vector
    timestep_cond = self.timestep_embedder(timestep)  # (B, hidden_size)

    # Process through DiT transformer
    transformer_output = self.updateformer(
        all_tokens,
        scene_tokens,
        timestep_cond  # Pass conditioning here!
    )

    # Extract outputs and predict noise
    track_output = transformer_output[:, :N, :, :]
    track_noise = self.track_head(track_output)

    outputs = {'track_noise': track_noise}

    if num_hands > 0:
        hand_output = transformer_output[:, N:, :, :]
        hand_noise = self.hand_head(hand_output)
        outputs['hand_noise'] = hand_noise

    return outputs
```

**Key changes:**
- No diffusion timestep in token building
- `timestep_embedder` produces conditioning vector
- Pass `timestep_cond` to transformer

---

## Phase 4: Update Training Configs

### 4.1 Update train_diffusion.yaml
**File:** `configs/train_diffusion.yaml`

**Changes:**
```yaml
# Model configuration
model:
  model_size: 'b'  # or 's', 'l'
  text_mode: true  # Enable text conditioning

# Curriculum learning - only affects ViT
unfreeze_after: 0.3  # Unfreeze ViT at 30% (text encoder always frozen)
vit_frozen: false    # ViT fine-tuning enabled
```

**Note:** Remove any SigLIP-specific configs.

### 4.2 Update dataset_config.yaml
**File:** `configs/dataset_config.yaml`

**No changes needed** - text is loaded as strings from H5.

---

## Phase 5: Backward Compatibility

### Decision: NO backward compatibility

**Reasoning:**
- Clean break for better architecture
- Old checkpoints will not load (expected)
- Users must retrain from scratch

**What to do:**
1. Add warning in train.py if loading old checkpoint
2. Update README with migration notes
3. Consider adding conversion script later if needed

**Checkpoint detection:**
```python
# In train.py
def load_checkpoint(checkpoint_path, model):
    ckpt = torch.load(checkpoint_path)
    state_dict = ckpt['model_state_dict']

    # Check if it's an old checkpoint
    if 'diffusion_time_emb' in state_dict or 'text_encoder.embeddings' in str(state_dict.keys()):
        raise ValueError(
            "Old checkpoint detected! This codebase uses DiT + UMT5 architecture. "
            "Old SigLIP + precomputed timestep checkpoints are incompatible. "
            "Please retrain from scratch or use an older version of the code."
        )

    model.load_state_dict(state_dict)
```

---

## Phase 6: Testing Plan

### 6.1 Unit Tests
Create `tests/test_dit_blocks.py`:
```python
def test_timestep_embedder():
    embedder = TimestepEmbedder(hidden_size=384)
    t = torch.randint(0, 1000, (8,))
    out = embedder(t)
    assert out.shape == (8, 384)

def test_dit_attn_block():
    block = DiTAttnBlock(hidden_size=384, num_heads=6)
    x = torch.randn(2, 64, 384)
    c = torch.randn(2, 384)
    out = block(x, c)
    assert out.shape == x.shape

def test_dit_updateformer():
    transformer = DiTUpdateFormer(
        depth=6, hidden_size=384, num_heads=6
    )
    all_tokens = torch.randn(2, 10, 5, 384)
    scene_tokens = torch.randn(2, 196, 384)
    timestep_cond = torch.randn(2, 384)
    out = transformer(all_tokens, scene_tokens, timestep_cond)
    assert out.shape == all_tokens.shape
```

### 6.2 Integration Test
```python
def test_diffusion_model_forward():
    model = DiffusionIntentTracker(
        model_size='s',
        text_mode=True,
        num_future_steps=10
    )

    frames = torch.randn(2, 1, 3, 256, 256)
    query_coords = torch.randn(2, 64, 2)
    noisy_traj = torch.randn(2, 64, 11, 2)
    timestep = torch.randint(0, 100, (2,))
    text = ["pick up the red cup", "place it on the table"]

    outputs = model(
        frames=frames,
        query_coords=query_coords,
        noisy_traj=noisy_traj,
        timestep=timestep,
        text=text
    )

    assert 'track_noise' in outputs
    assert outputs['track_noise'].shape == (2, 64, 11, 2)
```

### 6.3 Training Smoke Test
```bash
python train.py --config configs/train_diffusion.yaml --max_steps 100
```

---

## Phase 7: Implementation Order

### Step 1: Add DiT blocks (blocks.py)
1. `TimestepEmbedder`
2. `modulate` function
3. `DiTAttnBlock`
4. `DiTCrossAttnBlock`
5. `DiTUpdateFormer`

### Step 2: Update configs
1. `model_config.py` - UMT5 model names
2. `train_diffusion.yaml` - UMT5 config, `enable_cfg: false` by default
3. `train_diffusion_cfg.yaml` - Optional CFG config

### Step 3: Update base model
1. `base_model.py` - Replace SigLIP with UMT5
2. `base_model.py` - Add null embedding for CFG (always add, even if not used)
3. `base_model.py` - Add `_create_dit_updateformer()`
4. `base_model.py` - Update `encode_text()` with `use_null_embedding` parameter
5. Update freeze/unfreeze methods

### Step 4: Update diffusion model
1. `diffusion_model.py` - Replace timestep buffers with `TimestepEmbedder`
2. Update `_build_track_tokens` / `_build_hand_tokens`
3. Update `forward()` method with `cfg_dropout_prob` parameter
4. Update `compute_loss()` method with `cfg_dropout_prob` parameter
5. Update `predict()` method with `guidance_scale` parameter

### Step 5: Update training loop
1. `train.py` - Add CFG conditional logic based on `enable_cfg` config
2. Pass `cfg_dropout_prob` to compute_loss when enabled
3. Pass `guidance_scale` to predict during validation when enabled

### Step 6: Test
1. Run unit tests (without CFG)
2. Run integration test (without CFG)
3. Run training smoke test with `enable_cfg: false`
4. (Optional) Test with CFG enabled: `enable_cfg: true`

### Step 7: Clean up
1. Remove old code references
2. Update docstrings
3. Add checkpoint detection

---

## Expected Benefits

### 1. Better Diffusion Quality
- **Per-layer conditioning:** Timestep info propagates through all layers
- **Adaptive gates:** Network learns when to apply denoising
- **Proven architecture:** DiT is SOTA for diffusion models (validated by Large Video Planner)

### 2. Better Text Understanding
- **Instruction-tuned:** UMT5 understands imperative commands (vs SigLIP's captions)
- **Multilingual:** Supports 107 languages out of the box
- **Sequence output:** Full token sequence allows richer conditioning
- **Cross-attention preserved:** Scene features remain spatially grounded

### 3. Optional CFG for Better Instruction Following
- **Configurable:** Enable only when needed (`enable_cfg: true`)
- **No overhead when disabled:** Default behavior has no CFG cost
- **Tunable at inference:** Adjust guidance scale without retraining
- **Proven effective:** Large Video Planner uses CFG for robot control

### 4. Cleaner Architecture
- **Modular:** Timestep conditioning is explicit
- **Standard:** Follows established DiT pattern
- **Maintainable:** Clear separation of concerns

---

## Risks & Mitigations

### Risk 1: Training Instability
**Mitigation:** DiT uses zero-init gates, starts as identity network

### Risk 2: Memory Increase
**Mitigation:** Factorized attention preserved, UMT5 frozen (no gradients)

### Risk 3: Slower Inference
**Mitigation:**
- UMT5 encoding can be cached per instruction
- Timestep embedder is lightweight MLP

### Risk 4: Worse Performance Initially
**Mitigation:**
- Keep old codebase branch
- Retrain with same hyperparameters
- May need more training steps initially

---

## Success Criteria

1. ✅ Code runs without errors
2. ✅ Model trains stably (loss decreases)
3. ✅ Inference produces reasonable trajectories
4. ✅ Text conditioning affects predictions
5. ✅ Performance matches or exceeds old architecture (after sufficient training)

---

## Estimated Timeline

- **Phase 1:** 2-3 hours (Add DiT blocks to blocks.py)
- **Phase 2:** 1 hour (Update configs for UMT5)
- **Phase 3:** 2-3 hours (Add CFG infrastructure)
- **Phase 4:** 1-2 hours (Update DiffusionIntentTracker)
- **Phase 5:** 1 hour (Update training loop)
- **Phase 6:** 1-2 hours (Testing)
- **Phase 7:** 30 minutes (Cleanup)

**Total:** 8-12 hours of focused implementation

---

## Open Questions

1. **Should we add DiT's FinalLayer?** ✅ **Decision: No**
   - Current code uses simple linear heads
   - DiT uses conditioned final layer for image generation
   - Keep simple heads for trajectory prediction (different task)

2. **Token pooling for text?** ✅ **Decision: Use all tokens**
   - Use all UMT5 tokens (sequence output)
   - Richer conditioning than single pooled vector
   - Memory OK with frozen UMT5 (no gradients)

3. **CFG always enabled or optional?** ✅ **Decision: Optional via config**
   - Implemented as `enable_cfg: false` by default
   - Users opt-in when needed for instruction following
   - No overhead when disabled

4. **Should we use Diffusion Forcing?** ❓ **Decision: Future work**
   - Large Video Planner uses varying noise levels per timestep
   - Could improve temporal consistency
   - Add after validating base architecture
