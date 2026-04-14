"""
Cross-Request KV Caching
========================
Implements the KV caching mechanism from OneTrans.

Paper Reference: Section 3.5.1 (Cross Request KV Caching)

Two-stage paradigm:
  Stage I  (S-side, once per request):
    Process all S-tokens with causal masking and cache their K/V pairs
    and attention outputs. This stage executes ONCE per request.

  Stage II (NS-side, per candidate):
    For each candidate, compute its NS-tokens and perform cross-attention
    against the cached S-side K/V, followed by token-specific FFN layers.

Cross-request extension:
  Since user behavioral sequences are append-only, the cache from the
  previous request is reused and only incremental K/V for newly added
  behaviors are computed. This reduces per-request computation from
  O(L) to O(ΔL).
"""

import tensorflow as tf


class KVCache:
    """Cross-Request KV Cache for OneTrans.

    Manages hierarchical KV caches across layers and across requests.
    Supports incremental updates for append-only behavior sequences.

    Usage:
        cache = KVCache(num_layers=6)

        # First request: compute and cache all S-tokens
        cache.update_layer(layer_idx=0, k=K_0, v=V_0)
        ...

        # Next request: reuse cache, only compute new tokens
        cached_kv = cache.get_layer(layer_idx=0)
        # compute incremental K/V for new behaviors
        cache.append_layer(layer_idx=0, k_new=K_new, v_new=V_new)
    """

    def __init__(self, num_layers):
        """
        Args:
            num_layers: Number of OneTrans blocks (layers).
        """
        self.num_layers = num_layers
        self._cache = {}  # Dict[int, Tuple[tf.Tensor, tf.Tensor]]
        self._s_outputs = {}  # Cached S-side outputs per layer

    def reset(self):
        """Clear all cached K/V pairs (for new user session)."""
        self._cache.clear()
        self._s_outputs.clear()

    def has_cache(self, layer_idx):
        """Check if cache exists for a given layer."""
        return layer_idx in self._cache

    def get_layer(self, layer_idx):
        """Retrieve cached (K, V) for a specific layer.

        Args:
            layer_idx: Index of the OneTrans block.
        Returns:
            Tuple of (K_cache, V_cache) tensors, or None if not cached.
        """
        return self._cache.get(layer_idx, None)

    def update_layer(self, layer_idx, k, v):
        """Set or replace the K/V cache for a layer.

        Used in Stage I to cache S-token K/V.

        Args:
            layer_idx: Index of the OneTrans block.
            k: [B, L_cached, d] key tensor.
            v: [B, L_cached, d] value tensor.
        """
        self._cache[layer_idx] = (tf.identity(k), tf.identity(v))

    def append_layer(self, layer_idx, k_new, v_new):
        """Append new K/V to existing cache (cross-request incremental update).

        Used for append-only sequence updates, reducing computation from
        O(L) to O(ΔL).

        Args:
            layer_idx: Index of the OneTrans block.
            k_new: [B, ΔL, d] new key tensor for newly added behaviors.
            v_new: [B, ΔL, d] new value tensor.
        """
        if layer_idx in self._cache:
            k_old, v_old = self._cache[layer_idx]
            k_merged = tf.concat([k_old, k_new], axis=1)
            v_merged = tf.concat([v_old, v_new], axis=1)
            self._cache[layer_idx] = (k_merged, v_merged)
        else:
            self._cache[layer_idx] = (tf.identity(k_new), tf.identity(v_new))

    def set_s_output(self, layer_idx, s_output):
        """Cache S-side intermediate output for a layer.

        Args:
            layer_idx: Layer index.
            s_output: [B, L_S, d] S-token output after this layer.
        """
        self._s_outputs[layer_idx] = tf.identity(s_output)

    def get_s_output(self, layer_idx):
        """Retrieve cached S-side output.

        Args:
            layer_idx: Layer index.
        Returns:
            Cached S-token output tensor, or None.
        """
        return self._s_outputs.get(layer_idx, None)

    @property
    def cached_seq_length(self):
        """Return the current cached sequence length (from layer 0)."""
        if 0 in self._cache:
            return tf.shape(self._cache[0][0])[1]
        return 0


class TwoStageInference:
    """Two-Stage Inference Engine for OneTrans.

    Implements the two-stage KV caching paradigm from Sec 3.5.1:

    Stage I: Process S-tokens once per request.
      - Run S-tokens through all layers with causal masking.
      - Cache K/V pairs at each layer.

    Stage II: Process NS-tokens per candidate.
      - For each candidate item, compute its NS-tokens.
      - Cross-attend against cached S-side K/V.
      - Apply token-specific FFN layers.

    Args:
        model: The OneTrans model instance.
    """

    def __init__(self, model):
        self.model = model
        self.cache = KVCache(num_layers=model.config.num_layers)

    def reset(self):
        """Reset cache for a new user session."""
        self.cache.reset()

    def stage_one(self, s_tokens, s_mask=None):
        """Stage I: Process S-tokens and populate KV cache.

        This should be called ONCE per request.

        Args:
            s_tokens: [B, L_S, d] sequential tokens.
            s_mask: [B, L_S] optional mask.
        Returns:
            List of cached S-side outputs per layer.
        """
        self.cache.reset()

        x = s_tokens
        config = self.model.config
        l_s = tf.shape(s_tokens)[1]

        pyramid_schedule = self.model._get_pyramid_schedule(l_s)

        for i, block in enumerate(self.model.blocks):
            l_s_query = pyramid_schedule[i]

            # Only process S-tokens (no NS-tokens in Stage I)
            # We temporarily create a dummy input with just S-tokens
            # The block handles mixed parameterization internally

            # For S-only computation, we compute K/V over full S-tokens
            # and cache them
            x_normed = block.norm1(x)

            # Compute K, V for full S-token sequence using shared projections
            K = tf.matmul(x_normed, block.attention.shared_wk) + block.attention.shared_bk
            V = tf.matmul(x_normed, block.attention.shared_wv) + block.attention.shared_bv

            # Cache K/V at this layer
            self.cache.update_layer(i, K, V)

            # Compute Q for query subset
            if l_s_query < l_s:
                x_query = x[:, (l_s - l_s_query):, :]
                x_normed_q = block.norm1(x_query)
            else:
                x_normed_q = x_normed
                l_s_query = l_s

            Q = tf.matmul(x_normed_q, block.attention.shared_wq) + block.attention.shared_bq

            # Split heads and compute attention
            Q_h = block.attention._split_heads(Q)
            K_h = block.attention._split_heads(K)
            V_h = block.attention._split_heads(V)

            attn_scores = tf.matmul(Q_h, K_h, transpose_b=True) * block.attention.scale

            # Causal mask for S-only
            L_q = tf.shape(Q_h)[2]
            L_kv = tf.shape(K_h)[2]

            if l_s_query < l_s:
                q_pos = tf.range(l_s - l_s_query, l_s)
            else:
                q_pos = tf.range(l_s)

            kv_pos = tf.range(L_kv)
            causal_mask = tf.cast(
                q_pos[:, tf.newaxis] >= kv_pos[tf.newaxis, :],
                dtype=tf.float32,
            )
            attn_scores += (1.0 - causal_mask[tf.newaxis, tf.newaxis, :, :]) * (-1e9)
            attn_weights = tf.nn.softmax(attn_scores, axis=-1)
            attn_out = tf.matmul(attn_weights, V_h)
            attn_out = block.attention._merge_heads(attn_out)
            attn_out = tf.matmul(attn_out, block.attention.wo) + block.attention.bo

            # Residual
            if l_s_query < l_s:
                x_residual = x[:, (l_s - l_s_query):, :]
            else:
                x_residual = x
            z = attn_out + x_residual

            # FFN (shared for S-tokens only)
            z_normed = block.norm2(z)
            up = tf.matmul(z_normed, block.ffn.shared_w1) + block.ffn.shared_b1
            gate = tf.matmul(z_normed, block.ffn.shared_wg) + block.ffn.shared_bg
            hidden = tf.nn.silu(up) * gate
            ffn_out = tf.matmul(hidden, block.ffn.shared_w2) + block.ffn.shared_b2

            x = ffn_out + z

            # Update l_s for next layer
            l_s = l_s_query

            # Cache S-side output
            self.cache.set_s_output(i, x)

    def stage_two(self, ns_tokens):
        """Stage II: Process NS-tokens against cached S-side K/V.

        Called ONCE per candidate item.

        Args:
            ns_tokens: [B, L_NS, d] non-sequential tokens for this candidate.
        Returns:
            output: [B, L_NS, d] final NS-token representations.
        """
        x_ns = ns_tokens
        num_ns = tf.shape(ns_tokens)[1]

        for i, block in enumerate(self.model.blocks):
            # Get cached S-side K/V and output
            K_cached, V_cached = self.cache.get_layer(i)
            s_output = self.cache.get_s_output(i)

            # Norm NS-tokens
            x_ns_normed = block.norm1(x_ns)

            # NS-token-specific Q/K/V projections
            Q_ns = tf.einsum("bnd,ndo->bno", x_ns_normed,
                             block.attention.ns_wq) + block.attention.ns_bq
            K_ns = tf.einsum("bnd,ndo->bno", x_ns_normed,
                             block.attention.ns_wk) + block.attention.ns_bk
            V_ns = tf.einsum("bnd,ndo->bno", x_ns_normed,
                             block.attention.ns_wv) + block.attention.ns_bv

            # Concatenate cached S-side K/V with NS K/V
            K_all = tf.concat([K_cached, K_ns], axis=1)
            V_all = tf.concat([V_cached, V_ns], axis=1)

            # Multi-head attention
            Q_h = block.attention._split_heads(Q_ns)
            K_h = block.attention._split_heads(K_all)
            V_h = block.attention._split_heads(V_all)

            attn_scores = tf.matmul(Q_h, K_h, transpose_b=True) * block.attention.scale

            # NS-tokens can attend to all S-tokens and preceding NS-tokens
            L_kv = tf.shape(K_h)[2]
            L_s_cached = tf.shape(K_cached)[1]

            # All NS-tokens can attend to all S-tokens
            # NS-token i can attend to NS-tokens 0..i (causal among NS)
            ns_q_pos = tf.range(num_ns) + L_s_cached
            kv_pos = tf.range(L_kv)
            causal_mask = tf.cast(
                ns_q_pos[:, tf.newaxis] >= kv_pos[tf.newaxis, :],
                dtype=tf.float32,
            )
            attn_scores += (1.0 - causal_mask[tf.newaxis, tf.newaxis, :, :]) * (-1e9)

            attn_weights = tf.nn.softmax(attn_scores, axis=-1)
            attn_out = tf.matmul(attn_weights, V_h)
            attn_out = block.attention._merge_heads(attn_out)
            attn_out = tf.matmul(attn_out, block.attention.wo) + block.attention.bo

            z = attn_out + x_ns  # Residual

            # Token-specific FFN for NS-tokens
            z_normed = block.norm2(z)
            l_s_dummy = 0  # No S-tokens in this computation
            ffn_out = block.ffn(z_normed, l_s_dummy)

            x_ns = ffn_out + z  # Residual

        return x_ns
