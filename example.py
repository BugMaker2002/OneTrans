"""
OneTrans Example Usage
======================
Demonstrates how to build, train, and run inference with the OneTrans model.

This script provides:
  1. Synthetic data generation for testing
  2. Model construction (OneTransS and OneTransL configurations)
  3. Training with dual-optimizer strategy
  4. Two-stage inference with KV caching
  5. Next-batch evaluation protocol

Usage:
    python -m Multimodal.onetrans.example
"""

import tensorflow as tf
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Multimodal.onetrans.model import OneTrans, OneTransConfig, build_onetrans_s, build_onetrans_l
from Multimodal.onetrans.train import OneTransTrainer, compute_uauc
from Multimodal.onetrans.kv_cache import TwoStageInference


def create_synthetic_dataset(
    num_samples=1024,
    batch_size=64,
    ns_dim=128,
    seq_dims=(32, 32, 32),
    seq_lengths=(50, 40, 30),
    num_tasks=2,
    task_names=("ctr", "cvr"),
):
    """Create a synthetic dataset for testing OneTrans.

    Simulates:
      - Non-sequential features (user/item/context)
      - Multi-behavior sequences (e.g., purchase, add-to-cart, click)
      - Binary labels for CTR and CVR tasks

    Args:
        num_samples: Number of samples.
        batch_size: Batch size.
        ns_dim: Non-sequential feature dimension.
        seq_dims: Tuple of per-sequence embedding dimensions.
        seq_lengths: Tuple of max sequence lengths.
        num_tasks: Number of prediction tasks.
        task_names: Names of the tasks.
    Returns:
        tf.data.Dataset yielding (inputs, labels) dicts.
    """
    # Non-sequential features
    ns_features = np.random.randn(num_samples, ns_dim).astype(np.float32)

    # Multi-behavior sequences
    sequences = []
    seq_masks = []
    timestamps = []

    for i, (dim, length) in enumerate(zip(seq_dims, seq_lengths)):
        seq = np.random.randn(num_samples, length, dim).astype(np.float32)
        # Random valid lengths
        valid_lens = np.random.randint(length // 4, length + 1, size=num_samples)
        mask = np.zeros((num_samples, length), dtype=bool)
        ts = np.zeros((num_samples, length), dtype=np.int64)
        for j in range(num_samples):
            mask[j, :valid_lens[j]] = True
            ts[j, :valid_lens[j]] = np.sort(
                np.random.randint(1000000, 2000000, size=valid_lens[j])
            )
        sequences.append(seq)
        seq_masks.append(mask)
        timestamps.append(ts)

    # Labels
    labels = {}
    for task in task_names:
        labels[task] = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)

    # Build tf.data.Dataset
    def generator():
        for idx in range(num_samples):
            inputs = {
                "ns_features": ns_features[idx],
                "sequences": tuple(seq[idx] for seq in sequences),
                "seq_masks": tuple(m[idx] for m in seq_masks),
                "timestamps": tuple(t[idx] for t in timestamps),
            }
            label = {task: labels[task][idx] for task in task_names}
            yield inputs, label

    # Define output signature
    input_spec = {
        "ns_features": tf.TensorSpec(shape=(ns_dim,), dtype=tf.float32),
        "sequences": tuple(
            tf.TensorSpec(shape=(l, d), dtype=tf.float32)
            for l, d in zip(seq_lengths, seq_dims)
        ),
        "seq_masks": tuple(
            tf.TensorSpec(shape=(l,), dtype=tf.bool)
            for l in seq_lengths
        ),
        "timestamps": tuple(
            tf.TensorSpec(shape=(l,), dtype=tf.int64)
            for l in seq_lengths
        ),
    }
    label_spec = {task: tf.TensorSpec(shape=(1,), dtype=tf.float32) for task in task_names}

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(input_spec, label_spec),
    )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def example_basic_training():
    """Example 1: Basic training with OneTransS configuration."""
    print("=" * 60)
    print("Example 1: Basic Training (OneTransS)")
    print("=" * 60)

    # Model configuration
    config = OneTransConfig(
        d_model=64,          # Smaller for demo
        num_heads=4,
        num_layers=3,        # Fewer layers for demo
        d_ff=256,
        num_ns_tokens=4,
        ns_tokenizer_type="auto_split",
        ns_input_dim=128,
        seq_input_dims=[32, 32, 32],
        num_seq_types=3,
        max_seq_lengths=[50, 40, 30],
        merge_mode="timestamp_aware",
        pyramid_start_length=120,
        pyramid_end_length=4,
        task_names=["ctr", "cvr"],
        task_hidden_dim=64,
    )

    # Build model
    model = OneTrans(config)
    model.summary_info()

    # Create synthetic data
    train_dataset = create_synthetic_dataset(
        num_samples=256,
        batch_size=32,
        ns_dim=128,
        seq_dims=(32, 32, 32),
        seq_lengths=(50, 40, 30),
        task_names=("ctr", "cvr"),
    )

    val_dataset = create_synthetic_dataset(
        num_samples=64,
        batch_size=32,
        ns_dim=128,
        seq_dims=(32, 32, 32),
        seq_lengths=(50, 40, 30),
        task_names=("ctr", "cvr"),
    )

    # Create trainer
    trainer = OneTransTrainer(model, config={
        "dense_lr": 0.005,
        "sparse_lr": 0.01,
        "rms_alpha": 0.99999,
        "mixed_precision": False,
    })

    # Train
    history = trainer.train(
        train_dataset,
        val_dataset=val_dataset,
        epochs=3,
        verbose=True,
    )

    print("\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_metrics']:
        for k, v in history['val_metrics'][-1].items():
            print(f"Final val {k}: {v:.4f}")

    return model


def example_group_wise_tokenizer():
    """Example 2: Using Group-wise Tokenizer."""
    print("\n" + "=" * 60)
    print("Example 2: Group-wise Tokenizer")
    print("=" * 60)

    # Define feature groups
    group_dims = [32, 48, 24, 40]  # 4 feature groups

    config = OneTransConfig(
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        num_ns_tokens=len(group_dims),
        ns_tokenizer_type="group_wise",
        ns_group_dims=group_dims,
        seq_input_dims=[32, 32],
        num_seq_types=2,
        max_seq_lengths=[30, 20],
        merge_mode="timestamp_agnostic",
        pyramid_start_length=50,
        pyramid_end_length=len(group_dims),
        task_names=["ctr"],
        task_hidden_dim=64,
    )

    model = OneTrans(config)
    model.summary_info()

    # Create input with grouped features
    batch_size = 4
    group_features = [
        tf.random.normal((batch_size, dim)) for dim in group_dims
    ]

    sequences = [
        tf.random.normal((batch_size, 30, 32)),
        tf.random.normal((batch_size, 20, 32)),
    ]

    inputs = {
        "ns_features": group_features,
        "sequences": sequences,
    }

    # Forward pass
    outputs = model(inputs)
    print(f"CTR predictions shape: {outputs['ctr'].shape}")
    print(f"CTR predictions: {outputs['ctr'].numpy().flatten()}")


def example_two_stage_inference():
    """Example 3: Two-Stage Inference with KV Caching."""
    print("\n" + "=" * 60)
    print("Example 3: Two-Stage Inference (KV Caching)")
    print("=" * 60)

    config = OneTransConfig(
        d_model=64,
        num_heads=4,
        num_layers=3,
        d_ff=256,
        num_ns_tokens=4,
        ns_tokenizer_type="auto_split",
        ns_input_dim=128,
        seq_input_dims=[32, 32],
        num_seq_types=2,
        max_seq_lengths=[30, 20],
        merge_mode="timestamp_agnostic",
        pyramid_start_length=50,
        pyramid_end_length=4,
        task_names=["ctr", "cvr"],
        task_hidden_dim=64,
    )

    model = OneTrans(config)

    # Build model by running a forward pass
    batch_size = 2
    dummy_inputs = {
        "ns_features": tf.random.normal((batch_size, 128)),
        "sequences": [
            tf.random.normal((batch_size, 30, 32)),
            tf.random.normal((batch_size, 20, 32)),
        ],
    }
    _ = model(dummy_inputs)

    # Create two-stage inference engine
    engine = TwoStageInference(model)

    # Stage I: Process user behavior sequences (once per request)
    # Simulate tokenized S-tokens
    s_tokens = tf.random.normal((batch_size, 52, 64))  # 30+20+2sep tokens
    engine.stage_one(s_tokens)
    print("Stage I complete: S-tokens cached")

    # Stage II: Process each candidate item's NS-tokens
    num_candidates = 5
    for c in range(num_candidates):
        ns_tokens = tf.random.normal((batch_size, 4, 64))
        ns_output = engine.stage_two(ns_tokens)
        print(f"  Candidate {c}: NS-output shape = {ns_output.shape}")

    print("Two-stage inference complete!")


def example_scaling_comparison():
    """Example 4: Compare OneTransS vs OneTransL configurations."""
    print("\n" + "=" * 60)
    print("Example 4: Scaling Comparison (S vs L)")
    print("=" * 60)

    # Count parameters (excluding embeddings)
    def count_params(model):
        total = sum(np.prod(v.shape) for v in model.trainable_variables)
        return total

    # Build dummy input for model construction
    dummy_ns = tf.random.normal((2, 128))
    dummy_seqs = [tf.random.normal((2, 50, 32)) for _ in range(3)]
    dummy_input = {
        "ns_features": dummy_ns,
        "sequences": dummy_seqs,
    }

    # OneTransS
    model_s = build_onetrans_s(
        ns_input_dim=128,
        seq_input_dims=[32, 32, 32],
        max_seq_lengths=[50, 50, 50],
    )
    _ = model_s(dummy_input)
    params_s = count_params(model_s)
    print(f"OneTransS: {params_s:,} parameters ({params_s/1e6:.1f}M)")

    # OneTransL
    model_l = build_onetrans_l(
        ns_input_dim=128,
        seq_input_dims=[32, 32, 32],
        max_seq_lengths=[50, 50, 50],
    )
    dummy_ns_l = tf.random.normal((2, 128))
    dummy_input_l = {
        "ns_features": dummy_ns_l,
        "sequences": dummy_seqs,
    }
    _ = model_l(dummy_input_l)
    params_l = count_params(model_l)
    print(f"OneTransL: {params_l:,} parameters ({params_l/1e6:.1f}M)")

    print(f"Scale factor: {params_l / params_s:.1f}x")


def example_uauc_evaluation():
    """Example 5: UAUC Evaluation Metric."""
    print("\n" + "=" * 60)
    print("Example 5: UAUC Evaluation")
    print("=" * 60)

    # Simulate predictions from multiple users
    np.random.seed(42)
    num_users = 50
    impressions_per_user = 20

    user_ids = np.repeat(np.arange(num_users), impressions_per_user)
    labels = np.random.randint(0, 2, size=len(user_ids)).astype(np.float32)

    # Generate predictions that are somewhat correlated with labels
    noise = np.random.randn(len(user_ids)) * 0.3
    predictions = (labels + noise).clip(0, 1).astype(np.float32)

    uauc = compute_uauc(user_ids, labels, predictions)
    print(f"UAUC: {uauc:.4f}")

    # Compare with global AUC
    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(labels, predictions)
    global_auc = auc_metric.result().numpy()
    print(f"Global AUC: {global_auc:.4f}")


if __name__ == "__main__":
    print("OneTrans: Unified Feature Interaction and Sequence Modeling")
    print("with One Transformer in Industrial Recommender")
    print("Paper: arXiv:2510.26104 (WWW 2026)")
    print()

    # Run examples
    example_basic_training()
    example_group_wise_tokenizer()
    example_two_stage_inference()
    example_scaling_comparison()
    example_uauc_evaluation()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
