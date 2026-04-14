"""
OneTrans Block
==============
Implements the OneTrans Block with Pyramid Stack support.

Paper Reference:
  Section 3.3 (OneTrans Block) - Eq.4-5
  Section 3.4 (Pyramid Stack) - Eq.14

Each OneTrans block is a pre-norm causal Transformer block:
  Z^(n) = MixedMHA(Norm(X^(n-1))) + X^(n-1)    (Eq.4)
  X^(n) = MixedFFN(Norm(Z^(n)))   + Z^(n)       (Eq.5)

Pyramid Stack:
  At each layer, only a subset of the most recent S-tokens issue queries,
  while keys/values are computed over the full sequence. The query set
  shrinks with depth, progressively distilling information into NS-tokens.
"""

import tensorflow as tf
from Multimodal.onetrans.layers import RMSNorm, MixedCausalAttention, MixedFFN


class OneTransBlock(tf.keras.layers.Layer):
    """Single OneTrans Block with pyramid pruning support.

    Architecture (Fig.2b in paper):
      Pre-Norm -> MixedCausalAttention -> Residual
      Pre-Norm -> MixedFFN -> Residual

    With pyramid support:
      - At each layer, only the last L'_S S-tokens issue queries (Eq.14).
      - K/V are still computed over the full sequence.
      - After attention, only outputs for query positions are retained.

    Args:
        d_model: Model dimension (d).
        num_heads: Number of attention heads (H).
        d_ff: FFN hidden dimension.
        num_ns_tokens: Number of non-sequential tokens (L_NS).
        use_causal_mask: Whether to use causal attention mask.
    """

    def __init__(self, d_model, num_heads, d_ff, num_ns_tokens,
                 use_causal_mask=True, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_ns_tokens = num_ns_tokens

        # Pre-norm layers (RMSNorm, Sec 3.3)
        self.norm1 = RMSNorm(d_model, name="pre_norm_attn")
        self.norm2 = RMSNorm(d_model, name="pre_norm_ffn")

        # Mixed Causal Attention (Sec 3.3.1)
        self.attention = MixedCausalAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_ns_tokens=num_ns_tokens,
            use_causal_mask=use_causal_mask,
            name="mixed_causal_attn",
        )

        # Mixed FFN (Sec 3.3.2)
        self.ffn = MixedFFN(
            d_model=d_model,
            d_ff=d_ff,
            num_ns_tokens=num_ns_tokens,
            name="mixed_ffn",
        )

    def call(self, x, l_s, l_s_query=None, cached_kv=None, return_kv=False):
        """
        Args:
            x: [B, L, d] input token sequence (S-tokens followed by NS-tokens).
            l_s: int, total number of S-tokens in input (for K/V computation).
            l_s_query: int or None. If pyramid pruning is active, the number
                       of most recent S-tokens that issue queries. When None,
                       all S-tokens issue queries (no pruning).
            cached_kv: Optional (K_cache, V_cache) for cross-request caching.
            return_kv: Whether to return K/V for caching.
        Returns:
            output: [B, L', d] where L' = l_s_query + L_NS (if pyramid)
                    or [B, L, d] (if no pyramid).
            (optional) kv_cache tuple if return_kv=True.
        """
        if l_s_query is None:
            l_s_query = l_s

        # =============================================
        # Step 1: Pyramid - Select query tokens (Eq.14)
        # =============================================
        # Full sequence for K/V computation
        x_full = x  # [B, L, d]
        l_total = tf.shape(x)[1]

        # Query tokens: last l_s_query S-tokens + all NS-tokens
        if l_s_query < l_s:
            # Prune: keep only the tail l_s_query S-tokens as queries
            x_s_query = x[:, (l_s - l_s_query):l_s, :]  # [B, l_s_query, d]
            x_ns = x[:, l_s:, :]                          # [B, L_NS, d]
            x_query = tf.concat([x_s_query, x_ns], axis=1)  # [B, l_s_query + L_NS, d]
        else:
            x_query = x  # No pruning

        # =============================================
        # Step 2: Pre-Norm Attention (Eq.4)
        # Z^(n) = MixedMHA(Norm(X^(n-1))) + X^(n-1)
        # =============================================
        # Norm the full sequence for K/V
        x_normed_full = self.norm1(x_full)  # [B, L, d]

        if l_s_query < l_s:
            # For queries, extract the normed query positions
            x_normed_s_query = x_normed_full[:, (l_s - l_s_query):l_s, :]
            x_normed_ns = x_normed_full[:, l_s:, :]
            x_normed_query = tf.concat([x_normed_s_query, x_normed_ns], axis=1)

            # Compute K/V from full sequence, Q from query subset
            # We need to handle this by computing full K/V then selective Q
            attn_result = self._pyramid_attention(
                x_normed_query, x_normed_full, l_s, l_s_query,
                cached_kv=cached_kv, return_kv=return_kv,
            )
        else:
            attn_result = self.attention(
                x_normed_full, l_s,
                cached_kv=cached_kv, return_kv=return_kv,
            )

        if return_kv:
            attn_output, kv_cache = attn_result
        else:
            attn_output = attn_result

        # Residual connection
        z = attn_output + x_query  # [B, L', d]

        # =============================================
        # Step 3: Pre-Norm FFN (Eq.5)
        # X^(n) = MixedFFN(Norm(Z^(n))) + Z^(n)
        # =============================================
        z_normed = self.norm2(z)
        ffn_output = self.ffn(z_normed, l_s_query)  # [B, L', d]
        output = ffn_output + z  # [B, L', d]

        if return_kv:
            return output, kv_cache
        return output

    def _pyramid_attention(self, x_query, x_full, l_s, l_s_query,
                           cached_kv=None, return_kv=False):
        """Perform pyramid attention where Q comes from a subset of positions.

        Computes K/V over the full sequence but Q only from the query subset
        (last l_s_query S-tokens + all NS-tokens).

        This implements Eq.14:
          q_i = W_i^Q * x_i,  i in Q (query set)
          while K, V are computed over {1, ..., L}
        """
        # Compute K/V from full sequence
        K_full = self.attention._project_mixed(
            x_full, l_s,
            self.attention.shared_wk, self.attention.shared_bk,
            self.attention.ns_wk, self.attention.ns_bk,
        )
        V_full = self.attention._project_mixed(
            x_full, l_s,
            self.attention.shared_wv, self.attention.shared_bv,
            self.attention.ns_wv, self.attention.ns_bv,
        )

        # Compute Q from query subset
        Q_query = self.attention._project_mixed(
            x_query, l_s_query,
            self.attention.shared_wq, self.attention.shared_bq,
            self.attention.ns_wq, self.attention.ns_bq,
        )

        # Apply KV cache
        if cached_kv is not None:
            K_cache, V_cache = cached_kv
            K_full = tf.concat([K_cache, K_full], axis=1)
            V_full = tf.concat([V_cache, V_full], axis=1)

        # Split heads
        Q = self.attention._split_heads(Q_query)  # [B, H, L_q, head_dim]
        K = self.attention._split_heads(K_full)    # [B, H, L_kv, head_dim]
        V = self.attention._split_heads(V_full)    # [B, H, L_kv, head_dim]

        # Scaled dot-product attention with causal mask
        attn_scores = tf.matmul(Q, K, transpose_b=True) * self.attention.scale

        # Build causal mask for pyramid attention
        # Query positions map to the tail of S + all NS positions
        L_q = tf.shape(Q)[2]
        L_kv = tf.shape(K)[2]

        if cached_kv is not None:
            cache_len = tf.shape(cached_kv[0])[1]
        else:
            cache_len = 0

        # Query positions in the original sequence
        # S-query positions: [l_s - l_s_query, ..., l_s - 1]
        # NS-query positions: [l_s, ..., l_s + L_NS - 1]
        s_query_indices = tf.range(l_s - l_s_query, l_s)
        ns_query_indices = tf.range(l_s, l_s + self.num_ns_tokens)
        query_indices = tf.concat([s_query_indices, ns_query_indices], axis=0)
        query_indices = query_indices + cache_len  # Offset by cache

        kv_indices = tf.range(L_kv)

        # Causal: query at position i can attend to kv at position j where j <= i
        causal_mask = tf.cast(
            query_indices[:, tf.newaxis] >= kv_indices[tf.newaxis, :],
            dtype=tf.float32,
        )  # [L_q, L_kv]

        attn_scores = attn_scores + (1.0 - causal_mask[tf.newaxis, tf.newaxis, :, :]) * (-1e9)
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_output = tf.matmul(attn_weights, V)

        # Merge heads
        output = self.attention._merge_heads(attn_output)
        output = tf.matmul(output, self.attention.wo) + self.attention.bo

        if return_kv:
            K_full_merged = self.attention._merge_heads(K)
            V_full_merged = self.attention._merge_heads(V)
            return output, (K_full_merged, V_full_merged)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "num_ns_tokens": self.num_ns_tokens,
        })
        return config
