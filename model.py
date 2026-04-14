"""
OneTrans Model
==============
Full OneTrans model integrating all components.

Paper Reference: Section 3 (Methodology)

Architecture (Fig.2a):
  1. Unified Tokenizer:
     - Sequential features -> S-tokens via SequentialTokenizer
     - Non-sequential features -> NS-tokens via AutoSplit/GroupWise Tokenizer
  2. Concatenation: X^(0) = [S-tokens ; NS-tokens] (Eq.3)
  3. Pyramid-stacked OneTrans Blocks (Eq.4-5, 14)
  4. Final RMSNorm
  5. Task-specific prediction heads (CTR, CVR)

Hyperparameters (Sec 4.1.4):
  OneTransS: 6 layers, d=256, H=4, ~91M params
  OneTransL: 8 layers, d=384, H=4 (or more), ~330M params
"""

import tensorflow as tf
import math
from dataclasses import dataclass, field
from typing import List, Optional

from Multimodal.onetrans.layers import RMSNorm
from Multimodal.onetrans.block import OneTransBlock
from Multimodal.onetrans.tokenizer import AutoSplitTokenizer, GroupWiseTokenizer, SequentialTokenizer
from Multimodal.onetrans.kv_cache import KVCache


@dataclass
class OneTransConfig:
    """Configuration for OneTrans model.

    Attributes:
        d_model: Model dimension (d). Default 256 for OneTransS.
        num_heads: Number of attention heads (H). Default 4.
        num_layers: Number of stacked OneTrans blocks. Default 6 for OneTransS.
        d_ff: FFN hidden dimension. Default 4 * d_model.
        num_ns_tokens: Number of non-sequential tokens (L_NS). Default 12.
        ns_tokenizer_type: "auto_split" or "group_wise".
        ns_input_dim: Total NS feature embedding dimension.
        ns_group_dims: Feature group dimensions (for group_wise tokenizer).
        seq_input_dims: Per-sequence raw embedding dimensions.
        num_seq_types: Number of behavior sequence types.
        max_seq_lengths: Max length per sequence type.
        merge_mode: "timestamp_aware" or "timestamp_agnostic".
        pyramid_start_length: S-token query count at layer 0 (default 1190 for S).
        pyramid_end_length: S-token query count at final layer.
        task_names: List of task names (e.g., ["ctr", "cvr"]).
        task_hidden_dim: Hidden dim for task prediction heads.
        use_causal_mask: Whether to use causal attention (default True).
        grad_clip_dense: Gradient clipping threshold for dense layers (90).
        grad_clip_sparse: Gradient clipping threshold for sparse layers (120).
    """
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 6
    d_ff: int = 0  # Will be set to 4*d_model if 0
    num_ns_tokens: int = 12
    ns_tokenizer_type: str = "auto_split"
    ns_input_dim: int = 512
    ns_group_dims: Optional[List[int]] = None
    seq_input_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    num_seq_types: int = 3
    max_seq_lengths: List[int] = field(default_factory=lambda: [200, 200, 200])
    merge_mode: str = "timestamp_aware"
    pyramid_start_length: int = 1190
    pyramid_end_length: int = 0  # Will be set to num_ns_tokens if 0
    task_names: List[str] = field(default_factory=lambda: ["ctr", "cvr"])
    task_hidden_dim: int = 256
    use_causal_mask: bool = True
    grad_clip_dense: float = 90.0
    grad_clip_sparse: float = 120.0

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
        if self.pyramid_end_length == 0:
            self.pyramid_end_length = self.num_ns_tokens


class OneTrans(tf.keras.Model):
    """OneTrans: Unified Feature Interaction and Sequence Modeling
    with One Transformer in Industrial Recommender.

    Full model as described in the paper (Fig.2a).

    Args:
        config: OneTransConfig instance.
    """

    def __init__(self, config: OneTransConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # ============================================
        # 1. Unified Tokenizer (Sec 3.2)
        # ============================================
        # Non-Sequential Tokenizer
        if config.ns_tokenizer_type == "auto_split":
            self.ns_tokenizer = AutoSplitTokenizer(
                input_dim=config.ns_input_dim,
                d_model=config.d_model,
                num_ns_tokens=config.num_ns_tokens,
                name="auto_split_tokenizer",
            )
        elif config.ns_tokenizer_type == "group_wise":
            assert config.ns_group_dims is not None, \
                "ns_group_dims must be provided for group_wise tokenizer"
            self.ns_tokenizer = GroupWiseTokenizer(
                group_dims=config.ns_group_dims,
                d_model=config.d_model,
                name="group_wise_tokenizer",
            )
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {config.ns_tokenizer_type}")

        # Sequential Tokenizer
        self.seq_tokenizer = SequentialTokenizer(
            seq_input_dims=config.seq_input_dims,
            d_model=config.d_model,
            merge_mode=config.merge_mode,
            max_seq_lengths=config.max_seq_lengths,
            name="sequential_tokenizer",
        )

        # ============================================
        # 2. Stacked OneTrans Blocks (Sec 3.3)
        # ============================================
        self.blocks = []
        for i in range(config.num_layers):
            block = OneTransBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                num_ns_tokens=config.num_ns_tokens,
                use_causal_mask=config.use_causal_mask,
                name=f"onetrans_block_{i}",
            )
            self.blocks.append(block)

        # ============================================
        # 3. Final Normalization
        # ============================================
        self.final_norm = RMSNorm(config.d_model, name="final_norm")

        # ============================================
        # 4. Task-Specific Prediction Heads (Eq.2)
        # ============================================
        # Multi-task heads: CTR and CVR (or custom tasks)
        self.task_heads = {}
        for task_name in config.task_names:
            head = tf.keras.Sequential([
                tf.keras.layers.Dense(config.task_hidden_dim, activation="relu"),
                tf.keras.layers.Dense(config.task_hidden_dim // 2, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ], name=f"head_{task_name}")
            self.task_heads[task_name] = head

    def _get_pyramid_schedule(self, l_s):
        """Compute pyramid schedule for S-token query pruning.

        Paper Sec 4.1.4: "linearly shrinks the number of sequential query tokens
        from 1190 to 12 (OneTransS) / from 1500 to 16 (OneTransL)."
        "rounding the token count at each layer to the nearest multiple of 32"

        Args:
            l_s: int or tf.Tensor, actual number of S-tokens.
        Returns:
            List of int, the number of S-token queries at each layer.
        """
        num_layers = self.config.num_layers
        start = min(self.config.pyramid_start_length, int(l_s) if isinstance(l_s, int) else 10000)
        end = self.config.pyramid_end_length

        schedule = []
        for i in range(num_layers):
            if num_layers == 1:
                count = end
            else:
                # Linear interpolation from start to end
                ratio = i / (num_layers - 1)
                count = int(start * (1 - ratio) + end * ratio)

            # Round to nearest multiple of 32
            count = max(end, ((count + 16) // 32) * 32)
            if count < end:
                count = end
            schedule.append(count)

        return schedule

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: Dict with keys:
                - "ns_features": [B, ns_input_dim] or list of [B, g_i_dim]
                - "sequences": List of [B, L_i, dim_i] sequence tensors
                - "seq_masks": (optional) List of [B, L_i] boolean masks
                - "timestamps": (optional) List of [B, L_i] int64 timestamps
            training: bool, training mode flag.

        Returns:
            Dict of task predictions, e.g., {"ctr": [B, 1], "cvr": [B, 1]}.
        """
        # =============================================
        # Step 1: Tokenization (Sec 3.2)
        # =============================================
        # Non-Sequential Tokenization
        ns_features = inputs["ns_features"]
        ns_tokens = self.ns_tokenizer(ns_features)  # [B, L_NS, d]

        # Sequential Tokenization
        sequences = inputs["sequences"]
        seq_masks = inputs.get("seq_masks", None)
        timestamps = inputs.get("timestamps", None)

        s_tokens, s_mask, l_s = self.seq_tokenizer(
            sequences, seq_masks=seq_masks, timestamps=timestamps
        )

        # =============================================
        # Step 2: Concatenate tokens (Eq.3)
        # X^(0) = [S-tokens ; NS-tokens]
        # =============================================
        x = tf.concat([s_tokens, ns_tokens], axis=1)  # [B, L_S + L_NS, d]

        # =============================================
        # Step 3: Pyramid-stacked OneTrans Blocks (Eq.4-5, 14)
        # =============================================
        pyramid_schedule = self._get_pyramid_schedule(l_s)
        current_l_s = l_s

        for i, block in enumerate(self.blocks):
            l_s_query = min(pyramid_schedule[i], current_l_s)
            x = block(x, l_s=current_l_s, l_s_query=l_s_query)
            current_l_s = l_s_query  # Update S-token count after pruning

        # =============================================
        # Step 4: Final Normalization
        # =============================================
        x = self.final_norm(x)  # [B, L', d]

        # =============================================
        # Step 5: Extract NS-token representations for prediction
        # =============================================
        # After pyramid, x = [remaining S-tokens ; NS-tokens]
        # NS-tokens are at the tail
        ns_output = x[:, current_l_s:, :]  # [B, L_NS, d]

        # Pool NS-tokens for prediction (mean pooling)
        pooled = tf.reduce_mean(ns_output, axis=1)  # [B, d]

        # =============================================
        # Step 6: Task Predictions (Eq.2)
        # =============================================
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(pooled)  # [B, 1]

        return predictions

    def compute_loss(self, predictions, labels):
        """Compute multi-task binary cross-entropy loss.

        Args:
            predictions: Dict of {task_name: [B, 1] predicted probabilities}.
            labels: Dict of {task_name: [B, 1] binary labels}.
        Returns:
            total_loss: Scalar loss.
            task_losses: Dict of per-task losses.
        """
        task_losses = {}
        total_loss = 0.0

        for task_name in self.config.task_names:
            if task_name in predictions and task_name in labels:
                pred = predictions[task_name]
                label = tf.cast(labels[task_name], tf.float32)
                # Binary cross-entropy
                bce = tf.keras.losses.binary_crossentropy(
                    label, pred, from_logits=False
                )
                loss = tf.reduce_mean(bce)
                task_losses[task_name] = loss
                total_loss += loss

        return total_loss, task_losses

    def get_config(self):
        return {"config": self.config.__dict__}

    def summary_info(self):
        """Print model configuration summary."""
        c = self.config
        print("=" * 60)
        print("OneTrans Model Configuration")
        print("=" * 60)
        print(f"  d_model:          {c.d_model}")
        print(f"  num_heads:        {c.num_heads}")
        print(f"  num_layers:       {c.num_layers}")
        print(f"  d_ff:             {c.d_ff}")
        print(f"  num_ns_tokens:    {c.num_ns_tokens}")
        print(f"  ns_tokenizer:     {c.ns_tokenizer_type}")
        print(f"  seq_types:        {c.num_seq_types}")
        print(f"  merge_mode:       {c.merge_mode}")
        print(f"  task_names:       {c.task_names}")
        print(f"  pyramid_schedule: {c.pyramid_start_length} -> {c.pyramid_end_length}")
        print(f"  use_causal_mask:  {c.use_causal_mask}")
        print("=" * 60)


def build_onetrans_s(**kwargs):
    """Build OneTransS configuration (Sec 4.1.4).

    6 layers, d=256, H=4, ~91M parameters.
    Pyramid: 1190 -> 12 S-token queries across layers.
    """
    defaults = dict(
        d_model=256,
        num_heads=4,
        num_layers=6,
        d_ff=1024,
        num_ns_tokens=12,
        pyramid_start_length=1190,
        pyramid_end_length=12,
    )
    defaults.update(kwargs)
    config = OneTransConfig(**defaults)
    return OneTrans(config)


def build_onetrans_l(**kwargs):
    """Build OneTransL configuration (Sec 4.1.4).

    8 layers, d=384, ~330M parameters.
    Pyramid: 1500 -> 16 S-token queries across layers.
    """
    defaults = dict(
        d_model=384,
        num_heads=8,
        num_layers=8,
        d_ff=1536,
        num_ns_tokens=16,
        pyramid_start_length=1500,
        pyramid_end_length=16,
    )
    defaults.update(kwargs)
    config = OneTransConfig(**defaults)
    return OneTrans(config)
