"""
OneTrans Core Layers
====================
Implements:
  - RMSNorm (Root Mean Square Layer Normalization)
  - MixedCausalAttention (Mixed parameterization causal multi-head attention)
  - MixedFFN (Mixed parameterization feed-forward network)

Paper Reference: Section 3.3 (OneTrans Block)
  Eq.11-12: Mixed Q/K/V parameterization
  Eq.13:    Mixed FFN parameterization
"""

import tensorflow as tf
import math


class RMSNorm(tf.keras.layers.Layer):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Applied as pre-norm to all tokens, aligning scales across token types
    (S-tokens and NS-tokens) and stabilizing optimization.
    Paper Sec 3.3: "we apply RMSNorm as pre-norm to all tokens"
    """

    def __init__(self, d_model, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.eps = eps

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(self.d_model,),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x: [..., d_model]
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "eps": self.eps})
        return config


class MixedCausalAttention(tf.keras.layers.Layer):
    """Mixed (Shared / Token-Specific) Causal Multi-Head Attention.

    Paper Section 3.3.1, Eq.11-12:
      - S-tokens (i <= L_S) share a single set of Q/K/V projection weights.
      - Each NS-token (i > L_S) has its own token-specific Q/K/V projections.
      - Standard causal mask: each position attends only to preceding tokens.
      - NS-tokens attend to all S-tokens and preceding NS-tokens.

    Args:
        d_model: Model dimension (d in the paper).
        num_heads: Number of attention heads (H in the paper).
        num_ns_tokens: Number of non-sequential tokens (L_NS).
        use_causal_mask: Whether to apply causal masking (default True).
    """

    def __init__(self, d_model, num_heads, num_ns_tokens, use_causal_mask=True, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_ns_tokens = num_ns_tokens
        self.use_causal_mask = use_causal_mask
        self.scale = 1.0 / math.sqrt(self.head_dim)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    def build(self, input_shape):
        d = self.d_model

        # --- Shared Q/K/V for S-tokens (Eq.12: W_S^Psi) ---
        self.shared_wq = self.add_weight("shared_wq", shape=(d, d), initializer="glorot_uniform")
        self.shared_wk = self.add_weight("shared_wk", shape=(d, d), initializer="glorot_uniform")
        self.shared_wv = self.add_weight("shared_wv", shape=(d, d), initializer="glorot_uniform")
        self.shared_bq = self.add_weight("shared_bq", shape=(d,), initializer="zeros")
        self.shared_bk = self.add_weight("shared_bk", shape=(d,), initializer="zeros")
        self.shared_bv = self.add_weight("shared_bv", shape=(d,), initializer="zeros")

        # --- Token-specific Q/K/V for each NS-token (Eq.12: W_{NS,i}^Psi) ---
        ns = self.num_ns_tokens
        self.ns_wq = self.add_weight("ns_wq", shape=(ns, d, d), initializer="glorot_uniform")
        self.ns_wk = self.add_weight("ns_wk", shape=(ns, d, d), initializer="glorot_uniform")
        self.ns_wv = self.add_weight("ns_wv", shape=(ns, d, d), initializer="glorot_uniform")
        self.ns_bq = self.add_weight("ns_bq", shape=(ns, d), initializer="zeros")
        self.ns_bk = self.add_weight("ns_bk", shape=(ns, d), initializer="zeros")
        self.ns_bv = self.add_weight("ns_bv", shape=(ns, d), initializer="zeros")

        # --- Output projection ---
        self.wo = self.add_weight("wo", shape=(d, d), initializer="glorot_uniform")
        self.bo = self.add_weight("bo", shape=(d,), initializer="zeros")

        super().build(input_shape)

    def _project_mixed(self, x, l_s, shared_w, shared_b, ns_w, ns_b):
        """Apply mixed parameterization projection.

        Args:
            x: [B, L, d] input tensor
            l_s: number of S-tokens
            shared_w: [d, d] shared weight for S-tokens
            shared_b: [d] shared bias
            ns_w: [L_NS, d, d] token-specific weights for NS-tokens
            ns_b: [L_NS, d] token-specific biases
        Returns:
            projected: [B, L, d]
        """
        # S-tokens: shared projection (Eq.12 top)
        x_s = x[:, :l_s, :]  # [B, L_S, d]
        proj_s = tf.matmul(x_s, shared_w) + shared_b  # [B, L_S, d]

        # NS-tokens: token-specific projection (Eq.12 bottom)
        x_ns = x[:, l_s:, :]  # [B, L_NS, d]
        # Use einsum for efficient batched per-token projection
        # x_ns: [B, L_NS, d], ns_w: [L_NS, d, d] -> [B, L_NS, d]
        proj_ns = tf.einsum("bnd,ndo->bno", x_ns, ns_w) + ns_b  # [B, L_NS, d]

        return tf.concat([proj_s, proj_ns], axis=1)  # [B, L, d]

    def _split_heads(self, x):
        """Reshape [B, L, d] -> [B, H, L, head_dim]."""
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        x = tf.reshape(x, (B, L, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _merge_heads(self, x):
        """Reshape [B, H, L, head_dim] -> [B, L, d]."""
        B = tf.shape(x)[0]
        L = tf.shape(x)[2]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (B, L, self.d_model))

    def call(self, x, l_s, cached_kv=None, return_kv=False):
        """
        Args:
            x: [B, L, d] input token sequence.
            l_s: int, number of S-tokens in current input.
            cached_kv: optional (K_cache, V_cache) from previous computation.
            return_kv: whether to return (K, V) for caching.
        Returns:
            output: [B, L, d]
            (optional) (K, V) if return_kv=True
        """
        # Mixed Q/K/V projections (Eq.11)
        Q = self._project_mixed(x, l_s, self.shared_wq, self.shared_bq, self.ns_wq, self.ns_bq)
        K = self._project_mixed(x, l_s, self.shared_wk, self.shared_bk, self.ns_wk, self.ns_bk)
        V = self._project_mixed(x, l_s, self.shared_wv, self.shared_bv, self.ns_wv, self.ns_bv)

        # Apply KV cache if provided (Sec 3.5.1)
        if cached_kv is not None:
            K_cache, V_cache = cached_kv
            K = tf.concat([K_cache, K], axis=1)
            V = tf.concat([V_cache, V], axis=1)

        # Split into multi-heads
        Q = self._split_heads(Q)  # [B, H, L_q, head_dim]
        K = self._split_heads(K)  # [B, H, L_kv, head_dim]
        V = self._split_heads(V)  # [B, H, L_kv, head_dim]

        # Scaled dot-product attention
        attn_scores = tf.matmul(Q, K, transpose_b=True) * self.scale  # [B, H, L_q, L_kv]

        # Causal mask
        if self.use_causal_mask:
            L_q = tf.shape(Q)[2]
            L_kv = tf.shape(K)[2]
            # Build causal mask: position i can attend to positions <= i
            # When there's a cache, offset the query positions
            if cached_kv is not None:
                offset = tf.shape(cached_kv[0])[1]
            else:
                offset = 0
            # Row indices (query positions, offset by cache length)
            row_idx = tf.range(L_q) + offset  # [L_q]
            col_idx = tf.range(L_kv)  # [L_kv]
            causal_mask = tf.cast(
                row_idx[:, tf.newaxis] >= col_idx[tf.newaxis, :],
                dtype=tf.float32,
            )  # [L_q, L_kv]
            # Apply mask: set masked positions to -inf
            attn_scores = attn_scores + (1.0 - causal_mask[tf.newaxis, tf.newaxis, :, :]) * (-1e9)

        attn_weights = tf.nn.softmax(attn_scores, axis=-1)  # [B, H, L_q, L_kv]
        attn_output = tf.matmul(attn_weights, V)  # [B, H, L_q, head_dim]

        # Merge heads and output projection
        output = self._merge_heads(attn_output)  # [B, L_q, d]
        output = tf.matmul(output, self.wo) + self.bo

        if return_kv:
            # Return the full K, V before head splitting for caching
            K_full = self._merge_heads(K)
            V_full = self._merge_heads(V)
            return output, (K_full, V_full)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_ns_tokens": self.num_ns_tokens,
            "use_causal_mask": self.use_causal_mask,
        })
        return config


class MixedFFN(tf.keras.layers.Layer):
    """Mixed (Shared / Token-Specific) Feed-Forward Network.

    Paper Section 3.3.2, Eq.13:
      MixedFFN(x_i) = W_i^2 * phi(W_i^1 * x_i)

    Where W_i^1, W_i^2 follow the mixed parameterization of Eq.12:
      - Shared for S-tokens (i <= L_S)
      - Token-specific for NS-tokens (i > L_S)

    Uses SiLU (Swish) as activation function phi, following modern
    Transformer practices (consistent with LLM optimizations mentioned
    in the paper).

    Args:
        d_model: Model dimension.
        d_ff: Feed-forward inner dimension (typically 4 * d_model).
        num_ns_tokens: Number of non-sequential tokens.
    """

    def __init__(self, d_model, d_ff, num_ns_tokens, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_ns_tokens = num_ns_tokens

    def build(self, input_shape):
        d = self.d_model
        ff = self.d_ff
        ns = self.num_ns_tokens

        # --- Shared FFN for S-tokens ---
        self.shared_w1 = self.add_weight("shared_w1", shape=(d, ff), initializer="glorot_uniform")
        self.shared_b1 = self.add_weight("shared_b1", shape=(ff,), initializer="zeros")
        self.shared_w2 = self.add_weight("shared_w2", shape=(ff, d), initializer="glorot_uniform")
        self.shared_b2 = self.add_weight("shared_b2", shape=(d,), initializer="zeros")

        # Gate projection for SwiGLU variant (modern Transformer practice)
        self.shared_wg = self.add_weight("shared_wg", shape=(d, ff), initializer="glorot_uniform")
        self.shared_bg = self.add_weight("shared_bg", shape=(ff,), initializer="zeros")

        # --- Token-specific FFN for each NS-token ---
        self.ns_w1 = self.add_weight("ns_w1", shape=(ns, d, ff), initializer="glorot_uniform")
        self.ns_b1 = self.add_weight("ns_b1", shape=(ns, ff), initializer="zeros")
        self.ns_w2 = self.add_weight("ns_w2", shape=(ns, ff, d), initializer="glorot_uniform")
        self.ns_b2 = self.add_weight("ns_b2", shape=(ns, d), initializer="zeros")

        # Gate projection for NS-tokens
        self.ns_wg = self.add_weight("ns_wg", shape=(ns, d, ff), initializer="glorot_uniform")
        self.ns_bg = self.add_weight("ns_bg", shape=(ns, ff), initializer="zeros")

        super().build(input_shape)

    def call(self, x, l_s):
        """
        Args:
            x: [B, L, d] input tensor.
            l_s: int, number of S-tokens.
        Returns:
            output: [B, L, d]
        """
        # ---- S-tokens: shared FFN (Eq.13 with shared W) ----
        x_s = x[:, :l_s, :]  # [B, L_S, d]
        # SwiGLU: gate * silu(up)
        up_s = tf.matmul(x_s, self.shared_w1) + self.shared_b1   # [B, L_S, ff]
        gate_s = tf.matmul(x_s, self.shared_wg) + self.shared_bg  # [B, L_S, ff]
        hidden_s = tf.nn.silu(up_s) * gate_s                      # [B, L_S, ff]
        out_s = tf.matmul(hidden_s, self.shared_w2) + self.shared_b2  # [B, L_S, d]

        # ---- NS-tokens: token-specific FFN (Eq.13 with per-token W) ----
        x_ns = x[:, l_s:, :]  # [B, L_NS, d]
        # Per-token einsum projections
        up_ns = tf.einsum("bnd,ndf->bnf", x_ns, self.ns_w1) + self.ns_b1    # [B, L_NS, ff]
        gate_ns = tf.einsum("bnd,ndf->bnf", x_ns, self.ns_wg) + self.ns_bg  # [B, L_NS, ff]
        hidden_ns = tf.nn.silu(up_ns) * gate_ns                              # [B, L_NS, ff]
        out_ns = tf.einsum("bnf,nfd->bnd", hidden_ns, self.ns_w2) + self.ns_b2  # [B, L_NS, d]

        return tf.concat([out_s, out_ns], axis=1)  # [B, L, d]

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "num_ns_tokens": self.num_ns_tokens,
        })
        return config
