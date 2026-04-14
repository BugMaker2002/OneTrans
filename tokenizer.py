"""
OneTrans Tokenizers
===================
Implements:
  - AutoSplitTokenizer:  Non-Sequential tokenizer (Eq.7)
  - GroupWiseTokenizer:  Non-Sequential tokenizer (Eq.6)
  - SequentialTokenizer: Sequential tokenizer with merge strategies (Eq.8-10)

Paper Reference: Section 3.2 (Features and Tokenization)
"""

import tensorflow as tf


class AutoSplitTokenizer(tf.keras.layers.Layer):
    """Auto-Split Tokenizer for Non-Sequential Features (Eq.7).

    All non-sequential features are concatenated and projected once by a
    single MLP, then split into L_NS tokens of dimension d:

      NS-tokens = split(MLP(concat(NS)), L_NS)

    This reduces kernel launch overhead compared to Group-wise approach.

    Args:
        input_dim: Total concatenated dimension of all NS feature embeddings.
        d_model: Target token dimension (d).
        num_ns_tokens: Number of NS tokens to produce (L_NS).
        hidden_dim: MLP hidden dimension (default: 4 * d_model).
    """

    def __init__(self, input_dim, d_model, num_ns_tokens, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_ns_tokens = num_ns_tokens
        self.hidden_dim = hidden_dim or (4 * d_model)
        self.output_dim = num_ns_tokens * d_model

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
            tf.keras.layers.Dense(self.output_dim),
        ])

    def call(self, x):
        """
        Args:
            x: [B, input_dim] concatenated NS feature embeddings.
        Returns:
            ns_tokens: [B, L_NS, d] non-sequential tokens.
        """
        B = tf.shape(x)[0]
        projected = self.mlp(x)  # [B, L_NS * d]
        ns_tokens = tf.reshape(projected, (B, self.num_ns_tokens, self.d_model))
        return ns_tokens

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "num_ns_tokens": self.num_ns_tokens,
            "hidden_dim": self.hidden_dim,
        })
        return config


class GroupWiseTokenizer(tf.keras.layers.Layer):
    """Group-wise Tokenizer for Non-Sequential Features (Eq.6).

    Features are manually partitioned into semantic groups {g_1, ..., g_L_NS}.
    Each group is concatenated and passed through a group-specific MLP:

      NS-tokens = [MLP_1(concat(g_1)), ..., MLP_L_NS(concat(g_L_NS))]

    Args:
        group_dims: List of int, input dimension for each feature group.
        d_model: Target token dimension (d).
        hidden_dim: MLP hidden dimension per group (default: 2 * d_model).
    """

    def __init__(self, group_dims, d_model, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.group_dims = group_dims
        self.d_model = d_model
        self.num_ns_tokens = len(group_dims)
        self.hidden_dim = hidden_dim or (2 * d_model)

        # One MLP per group (Eq.6: MLP_i for group g_i)
        self.group_mlps = []
        for i, gdim in enumerate(group_dims):
            mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ], name=f"group_mlp_{i}")
            self.group_mlps.append(mlp)

    def call(self, group_features):
        """
        Args:
            group_features: List of [B, g_i_dim] tensors, one per feature group.
        Returns:
            ns_tokens: [B, L_NS, d] non-sequential tokens.
        """
        tokens = []
        for i, (feat, mlp) in enumerate(zip(group_features, self.group_mlps)):
            token = mlp(feat)  # [B, d]
            tokens.append(token)
        # Stack: [B, L_NS, d]
        return tf.stack(tokens, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "group_dims": self.group_dims,
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
        })
        return config


class SequentialTokenizer(tf.keras.layers.Layer):
    """Sequential Tokenizer for User Behavior Sequences (Eq.8-10).

    Accepts multi-behavior sequences S = {S_1, ..., S_n}.
    Each sequence S_i has events [e_{i,1}, ..., e_{i,L_i}], where each
    event embedding e is constructed by concatenating item ID with side info.

    A shared per-sequence MLP_i projects events to common dimension d (Eq.9).
    Sequences are merged into a single token sequence (Eq.10) using either:
      - timestamp_aware: interleave events by time with type indicators
      - timestamp_agnostic: concatenate by intent order with [SEP] tokens

    Args:
        seq_input_dims: List of int, raw embedding dim for each sequence type.
        d_model: Target token dimension (d).
        merge_mode: "timestamp_aware" or "timestamp_agnostic".
        max_seq_lengths: List of int, max length for each sequence type.
    """

    def __init__(self, seq_input_dims, d_model, merge_mode="timestamp_aware",
                 max_seq_lengths=None, **kwargs):
        super().__init__(**kwargs)
        self.seq_input_dims = seq_input_dims
        self.d_model = d_model
        self.num_seq_types = len(seq_input_dims)
        self.merge_mode = merge_mode
        self.max_seq_lengths = max_seq_lengths or [100] * self.num_seq_types

    def build(self, input_shape):
        # Per-sequence projection MLPs (Eq.9: MLP_i)
        self.seq_mlps = []
        for i, dim in enumerate(self.seq_input_dims):
            mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(self.d_model, activation="relu"),
                tf.keras.layers.Dense(self.d_model),
            ], name=f"seq_mlp_{i}")
            # Build the MLP eagerly
            mlp.build((None, dim))
            self.seq_mlps.append(mlp)

        # Learnable [SEP] tokens for sequence boundaries (Sec 3.2.2)
        # One [SEP] per sequence boundary: n-1 boundaries for n sequences
        # + 1 between S-tokens and NS-tokens = n [SEP] tokens total
        self.sep_tokens = self.add_weight(
            name="sep_tokens",
            shape=(self.num_seq_types, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Sequence type embeddings for timestamp-aware mode
        self.seq_type_embeddings = self.add_weight(
            name="seq_type_embeddings",
            shape=(self.num_seq_types, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Learnable position embeddings for sequence positions
        total_max_len = sum(self.max_seq_lengths) + self.num_seq_types
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(total_max_len, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, sequences, seq_masks=None, timestamps=None):
        """
        Args:
            sequences: List of [B, L_i, dim_i] tensors, one per sequence type.
            seq_masks: List of [B, L_i] boolean masks (True = valid event).
            timestamps: List of [B, L_i] int64 timestamps (for timestamp_aware mode).
        Returns:
            s_tokens: [B, L_S, d] sequential token sequence.
            s_mask: [B, L_S] boolean mask for valid tokens.
            l_s: int, total length of S-tokens.
        """
        B = tf.shape(sequences[0])[0]

        # Project each sequence to common dimension d (Eq.9)
        projected_seqs = []
        for i, (seq, mlp) in enumerate(zip(sequences, self.seq_mlps)):
            proj = mlp(seq)  # [B, L_i, d]
            # Add sequence type embedding
            proj = proj + self.seq_type_embeddings[i:i+1, :]
            projected_seqs.append(proj)

        if self.merge_mode == "timestamp_aware" and timestamps is not None:
            return self._merge_timestamp_aware(
                B, projected_seqs, seq_masks, timestamps
            )
        else:
            return self._merge_timestamp_agnostic(
                B, projected_seqs, seq_masks
            )

    def _merge_timestamp_agnostic(self, B, projected_seqs, seq_masks):
        """Concatenate sequences by event impact with [SEP] tokens.

        Order: purchase -> add-to-cart -> click (sequences should be ordered
        by user intent, with highest intent first).
        """
        all_tokens = []
        all_masks = []

        for i, proj_seq in enumerate(projected_seqs):
            # Add projected sequence tokens
            all_tokens.append(proj_seq)  # [B, L_i, d]
            if seq_masks is not None:
                all_masks.append(seq_masks[i])
            else:
                all_masks.append(tf.ones(tf.shape(proj_seq)[:2], dtype=tf.bool))

            # Insert [SEP] token after each sequence
            sep = tf.broadcast_to(
                self.sep_tokens[i:i+1, :],
                (B, 1, self.d_model)
            )
            all_tokens.append(sep)
            all_masks.append(tf.ones((B, 1), dtype=tf.bool))

        s_tokens = tf.concat(all_tokens, axis=1)  # [B, L_S, d]
        s_mask = tf.concat(all_masks, axis=1)      # [B, L_S]

        # Add positional embeddings
        l_s = tf.shape(s_tokens)[1]
        pos_emb = self.position_embeddings[:l_s, :]  # [L_S, d]
        s_tokens = s_tokens + pos_emb[tf.newaxis, :, :]

        return s_tokens, s_mask, l_s

    def _merge_timestamp_aware(self, B, projected_seqs, seq_masks, timestamps):
        """Interleave events by timestamp with sequence-type indicators.

        Events from different sequences are sorted by timestamp.
        """
        # Flatten all events with their timestamps and type indices
        all_events_list = []
        all_times_list = []
        all_types_list = []
        all_masks_list = []

        for i, (proj_seq, ts) in enumerate(zip(projected_seqs, timestamps)):
            L_i = tf.shape(proj_seq)[1]
            all_events_list.append(proj_seq)                           # [B, L_i, d]
            all_times_list.append(ts)                                  # [B, L_i]
            all_types_list.append(tf.fill((B, L_i), i))               # [B, L_i]
            if seq_masks is not None:
                all_masks_list.append(seq_masks[i])
            else:
                all_masks_list.append(tf.ones((B, L_i), dtype=tf.bool))

        all_events = tf.concat(all_events_list, axis=1)  # [B, L_total, d]
        all_times = tf.concat(all_times_list, axis=1)      # [B, L_total]
        all_masks = tf.concat(all_masks_list, axis=1)      # [B, L_total]

        # Sort by timestamp (per sample in batch)
        # Set invalid timestamps to a large value so they sort to the end
        large_val = tf.constant(2**31 - 1, dtype=all_times.dtype)
        sort_times = tf.where(all_masks, all_times, large_val)
        sort_indices = tf.argsort(sort_times, axis=1, stable=True)  # [B, L_total]

        # Gather sorted events
        s_tokens = tf.gather(all_events, sort_indices, batch_dims=1)  # [B, L_total, d]
        s_mask = tf.gather(all_masks, sort_indices, batch_dims=1)      # [B, L_total]

        # Add positional embeddings
        l_s = tf.shape(s_tokens)[1]
        pos_emb = self.position_embeddings[:l_s, :]
        s_tokens = s_tokens + pos_emb[tf.newaxis, :, :]

        return s_tokens, s_mask, l_s

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_input_dims": self.seq_input_dims,
            "d_model": self.d_model,
            "merge_mode": self.merge_mode,
            "max_seq_lengths": self.max_seq_lengths,
        })
        return config
