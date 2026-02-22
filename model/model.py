"""GPT-2 style decoder-only transformer for ALS-LM.

A clean standalone implementation of a GPT-2 decoder-only transformer,
inspired by nanoGPT's architectural choices but written from scratch as a
portfolio-quality module for the ALS-LM project.

Key design choices:

- **Pre-LN architecture:** Layer normalization is applied before each sublayer
  (attention and MLP), not after. This provides more stable training and better
  gradient flow compared to the original Post-LN transformer, especially at
  depth. Pre-LN tolerates higher learning rates and converges ~40% faster in
  wall-time.

- **nn.Linear layers:** All projections use standard PyTorch ``nn.Linear``
  rather than HuggingFace's ``Conv1D``. This is idiomatic PyTorch and easier
  to debug. The weight transposition required by HuggingFace GPT2LMHeadModel
  is handled in the export/conversion script (``export/convert_to_hf.py``).

- **Weight tying:** The token embedding matrix (``transformer.wte``) and the
  language model head (``lm_head``) share the same weight tensor, following
  the standard GPT-2 convention. This reduces parameters and ensures input
  and output token representations live in the same vector space.

- **Flash Attention:** Uses PyTorch's ``F.scaled_dot_product_attention`` with
  ``is_causal=True``, which automatically dispatches to FlashAttention on
  Ampere GPUs (RTX 3060, SM 8.6) for O(N) memory and ~2x speedup.

- **Learned positional embeddings:** Standard GPT-2 positional embeddings
  (up to ``block_size`` positions), not RoPE.

Three named configurations are provided:

- ``tiny`` (~9M params): 6 layers, 6 heads, 192 embed dim. For pipeline
  validation and fast iteration during development.
- ``medium`` (~111M params): 12 layers, 12 heads, 768 embed dim. Matches
  GPT-2 Small dimensions for intermediate testing.
- ``500M`` (~516M params): 24 layers, 16 heads, 1280 embed dim. Production
  training target for the ALS domain model.

All named configs set ``vocab_size=None`` as a sentinel value. The actual
vocabulary size must be supplied at model construction time (typically read
from ``data/tokenized/meta.pkl``), ensuring the model always matches the
tokenizer.

Usage::

    from model.model import GPT

    # Create a tiny model for testing
    model = GPT.from_config("tiny", vocab_size=32768)

    # Forward pass
    logits, loss = model(input_ids, targets=target_ids)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Configuration for a GPT-2 style decoder-only transformer.

    Attributes:
        block_size: Maximum sequence length (number of positions). The model
            learns one positional embedding per position, so inputs longer
            than ``block_size`` will raise an error.
        vocab_size: Number of tokens in the vocabulary. This determines the
            size of the token embedding table and the language model head.
            Set to ``None`` in named configs as a sentinel; must be overridden
            from ``meta.pkl`` at runtime.
        n_layer: Number of transformer blocks (depth of the model).
        n_head: Number of attention heads per block. Must evenly divide
            ``n_embd`` so each head has dimension ``n_embd // n_head``.
        n_embd: Embedding dimension (model width). All residual stream
            vectors, attention outputs, and MLP intermediates use this
            dimension (MLP expands to 4x internally).
        dropout: Dropout probability applied to attention weights, residual
            connections, and embedding outputs. Set to 0.0 during inference.
        bias: Whether to include bias terms in Linear layers and LayerNorms.
            GPT-2 uses bias=True. Setting to False saves a small number of
            parameters and is sometimes preferred for modern models.
    """

    block_size: int = 1024
    vocab_size: Optional[int] = 32768
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


# Named configurations for the ALS-LM project.
# vocab_size is None (sentinel): it MUST be overridden from meta.pkl at
# runtime. The model constructor raises ValueError if vocab_size is None.
MODEL_CONFIGS: dict[str, GPTConfig] = {
    "tiny": GPTConfig(
        block_size=1024,
        vocab_size=None,
        n_layer=6,
        n_head=6,
        n_embd=192,
        dropout=0.0,
        bias=True,
    ),
    "medium": GPTConfig(
        block_size=1024,
        vocab_size=None,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    ),
    "500M": GPTConfig(
        block_size=1024,
        vocab_size=None,
        n_layer=24,
        n_head=16,
        n_embd=1280,
        dropout=0.0,
        bias=True,
    ),
}


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention support.

    Computes scaled dot-product attention over all heads in parallel using a
    combined QKV projection. The causal mask ensures each position can only
    attend to itself and earlier positions, enforcing the autoregressive
    property required for language modeling.

    The combined QKV projection (``c_attn``) computes queries, keys, and
    values in a single matrix multiply, which is more efficient than three
    separate projections. The output projection (``c_proj``) maps the
    concatenated head outputs back to the residual stream dimension.

    Flash Attention (via ``F.scaled_dot_product_attention``) is used for the
    attention computation, providing O(N) memory usage and hardware-efficient
    fused kernels on Ampere GPUs.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )
        # Combined query, key, value projection: (n_embd) -> (3 * n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection: (n_embd) -> (n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim
        head_dim = C // self.n_head

        # Compute Q, K, V for all heads in one projection, then split
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape from (B, T, C) to (B, n_head, T, head_dim) for multi-head
        # attention. The transpose(1, 2) swaps the sequence and head dims so
        # that each head operates on its own (T, head_dim) slice independently.
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask.
        # PyTorch 2.x dispatches to FlashAttention on supported hardware
        # (Ampere GPUs), providing O(N) memory and fused CUDA kernels.
        dropout_p = self.attn_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        # Reassemble all head outputs: (B, n_head, T, head_dim) -> (B, T, C).
        # contiguous() is needed after transpose to ensure memory layout is
        # compatible with the subsequent view() reshape.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection and residual dropout
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward network with GELU activation and 4x expansion ratio.

    The MLP in each transformer block expands the embedding dimension by 4x,
    applies a GELU nonlinearity, then projects back down. This expansion
    provides the model's primary nonlinear transformation capacity.

    GELU (Gaussian Error Linear Unit) is the activation used in the original
    GPT-2. PyTorch's ``nn.GELU()`` with default ``approximate='none'``
    matches the original implementation exactly.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # Up-projection: (n_embd) -> (4 * n_embd)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        # Down-projection: (4 * n_embd) -> (n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with Pre-LN (layer norm before sublayers).

    Pre-LN applies layer normalization before each sublayer (attention and
    MLP), then adds the sublayer output as a residual:

        x = x + attn(ln_1(x))
        x = x + mlp(ln_2(x))

    This differs from the original transformer's Post-LN design, which
    normalizes after the residual addition. Pre-LN provides several
    advantages for training:

    - More stable gradients, especially in deeper networks (24+ layers)
    - Tolerance for higher learning rates without divergence
    - ~40% faster convergence in wall-time compared to Post-LN

    The trade-off is slightly growing hidden state variance across layers,
    which is managed by the final layer norm (``ln_f``) in the GPT class.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: normalize before each sublayer, add residual after
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2 style decoder-only transformer language model.

    A complete autoregressive language model consisting of token and position
    embeddings, a stack of transformer blocks (each with causal self-attention
    and an MLP), a final layer norm, and a language model head that projects
    to vocabulary logits.

    The model uses weight tying between the token embedding (``wte``) and the
    language model head (``lm_head``), sharing the same weight matrix for both
    input token lookup and output token prediction.

    Weight initialization follows the GPT-2 convention: all Linear and
    Embedding weights are initialized from N(0, 0.02), biases are zeroed, and
    residual projection layers (``c_proj``) use a scaled initialization of
    N(0, 0.02 / sqrt(2 * n_layer)) to prevent residual stream variance from
    growing with depth.

    Args:
        config: A ``GPTConfig`` instance specifying the model architecture.
            The ``vocab_size`` must not be ``None``; use ``from_config()`` to
            create models from named configurations with proper vocab_size
            override.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.vocab_size is None:
            raise ValueError(
                "vocab_size must not be None. Named configs use None as a "
                "sentinel; override it with the actual vocabulary size from "
                "meta.pkl. Use GPT.from_config('tiny', vocab_size=32768)."
            )
        assert config.block_size is not None and config.block_size > 0
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the token embedding and LM head share the same tensor.
        # This must be set AFTER both modules are created so that the shared
        # tensor is properly registered. The embedding lookup and output
        # projection then operate in the same vector space.
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Apply scaled initialization to residual projections (c_proj layers).
        # The scaling factor 1/sqrt(2*n_layer) prevents the residual stream
        # variance from growing proportionally with depth (GPT-2 paper).
        self._init_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with N(0, 0.02) for Linear/Embedding, zeros for biases."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _init_residual_projections(self) -> None:
        """Apply scaled initialization to residual projection layers.

        The c_proj layers in both attention and MLP are the "output" projections
        that feed into the residual stream. Scaling their initialization by
        1/sqrt(2*n_layer) compensates for the accumulation of residuals across
        layers, preventing the variance of the hidden states from growing with
        model depth.
        """
        scale = 1.0 / math.sqrt(2 * self.config.n_layer)
        for block in self.transformer.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.02 * scale)
            torch.nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=0.02 * scale)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run a forward pass through the model.

        Args:
            idx: Input token indices of shape ``(B, T)`` where B is batch size
                and T is sequence length. T must be <= ``config.block_size``.
            targets: Optional target token indices of shape ``(B, T)`` for
                computing cross-entropy loss. If ``None``, loss is not computed.

        Returns:
            A tuple ``(logits, loss)`` where logits has shape ``(B, T, V)``
            (V = vocab_size) and loss is a scalar tensor if targets were
            provided, or ``None`` otherwise.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # Create position indices [0, 1, ..., T-1] on the same device as input
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Token and position embeddings, summed and passed through dropout
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd) -> broadcast to (B, T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm and projection to vocabulary logits
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute cross-entropy loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Count the number of parameters in the model.

        Args:
            non_embedding: If True, exclude positional embedding parameters.
                Position embeddings do not scale with model depth, so excluding
                them gives a better measure of the transformer's "size" for
                comparing configurations with different block sizes.

        Returns:
            Total number of parameters (int).
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @classmethod
    def from_config(
        cls,
        config_name: str,
        vocab_size: Optional[int] = None,
        **overrides,
    ) -> "GPT":
        """Create a model from a named configuration.

        This is the primary way to instantiate models. It looks up a named
        configuration from ``MODEL_CONFIGS``, applies the required
        ``vocab_size`` override, applies any additional keyword overrides, and
        returns the instantiated model.

        Args:
            config_name: Name of the configuration to use. Must be one of
                the keys in ``MODEL_CONFIGS`` (currently "tiny", "medium",
                "500M").
            vocab_size: Vocabulary size to use. Required because named configs
                set ``vocab_size=None`` as a sentinel.
            **overrides: Additional configuration overrides (e.g.,
                ``dropout=0.1``, ``block_size=512``).

        Returns:
            An instantiated ``GPT`` model.

        Raises:
            KeyError: If ``config_name`` is not in ``MODEL_CONFIGS``.
            ValueError: If ``vocab_size`` is not provided and the named config
                has ``vocab_size=None``.

        Example::

            model = GPT.from_config("tiny", vocab_size=32768)
            model = GPT.from_config("medium", vocab_size=32768, dropout=0.1)
        """
        if config_name not in MODEL_CONFIGS:
            raise KeyError(
                f"Unknown config '{config_name}'. "
                f"Available: {list(MODEL_CONFIGS.keys())}"
            )

        # Start from the named config template
        base_config = MODEL_CONFIGS[config_name]

        # Build a dict of all config fields, applying overrides
        config_dict = {
            "block_size": base_config.block_size,
            "vocab_size": base_config.vocab_size,
            "n_layer": base_config.n_layer,
            "n_head": base_config.n_head,
            "n_embd": base_config.n_embd,
            "dropout": base_config.dropout,
            "bias": base_config.bias,
        }

        # Apply vocab_size override (required for named configs)
        if vocab_size is not None:
            config_dict["vocab_size"] = vocab_size

        # Apply any additional overrides
        config_dict.update(overrides)

        config = GPTConfig(**config_dict)
        return cls(config)
