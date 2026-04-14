"""
OneTrans Training Script
========================
Implements the training loop with dual-optimizer strategy and
next-batch evaluation protocol as described in the paper.

Paper Reference: Section 4.1.4 (Hyperparameter Settings)

Dual-optimizer strategy (without weight decay):
  - Sparse embeddings: Adagrad (beta1=0.1, beta2=1.0)
  - Dense parameters: RMSProp (lr=0.005, alpha=0.99999, momentum=0)

Stabilization techniques:
  - Pre-Norm (built into OneTrans blocks)
  - Global grad-norm clipping (90 for dense, 120 for sparse)
  - Mixed precision training (BF16/FP16)

Evaluation:
  - Next-batch evaluation protocol
  - AUC and UAUC metrics
"""

import tensorflow as tf
import numpy as np
import time
from typing import Dict, List, Optional, Tuple


class DualOptimizer:
    """Dual-optimizer strategy from Sec 4.1.4.

    Sparse embeddings use Adagrad; dense parameters use RMSProp.

    Args:
        model: OneTrans model instance.
        dense_lr: Learning rate for dense parameters (default: 0.005).
        sparse_lr: Learning rate for sparse embeddings (default: 0.01).
        rms_alpha: RMSProp alpha/decay (default: 0.99999).
        grad_clip_dense: Gradient clip norm for dense (default: 90).
        grad_clip_sparse: Gradient clip norm for sparse (default: 120).
    """

    def __init__(self, model, dense_lr=0.005, sparse_lr=0.01,
                 rms_alpha=0.99999, grad_clip_dense=90.0,
                 grad_clip_sparse=120.0):
        self.model = model
        self.grad_clip_dense = grad_clip_dense
        self.grad_clip_sparse = grad_clip_sparse

        # Dense optimizer: RMSProp (Sec 4.1.4)
        # Paper: lr=0.005, alpha=0.99999, momentum=0
        self.dense_optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=dense_lr,
            rho=rms_alpha,  # alpha in paper = rho in TF
            momentum=0.0,
        )

        # Sparse optimizer: Adagrad (Sec 4.1.4)
        # Paper: beta1=0.1, beta2=1.0 (Adagrad-like)
        self.sparse_optimizer = tf.keras.optimizers.Adagrad(
            learning_rate=sparse_lr,
            initial_accumulator_value=0.1,
        )

    def _partition_variables(self):
        """Partition model variables into sparse (embeddings) and dense."""
        sparse_vars = []
        dense_vars = []

        for var in self.model.trainable_variables:
            name = var.name.lower()
            # Heuristic: embedding tables and tokenizer embeddings are "sparse"
            if ("embedding" in name or "sep_token" in name or
                    "position_embedding" in name or "seq_type_embedding" in name):
                sparse_vars.append(var)
            else:
                dense_vars.append(var)

        return sparse_vars, dense_vars

    @tf.function
    def train_step(self, inputs, labels):
        """Execute one training step.

        Args:
            inputs: Dict of input features.
            labels: Dict of {task_name: [B, 1] labels}.
        Returns:
            total_loss: Scalar.
            task_losses: Dict of per-task losses.
        """
        sparse_vars, dense_vars = self._partition_variables()
        all_vars = dense_vars + sparse_vars

        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            total_loss, task_losses = self.model.compute_loss(predictions, labels)

        gradients = tape.gradient(total_loss, all_vars)

        # Split gradients
        n_dense = len(dense_vars)
        dense_grads = gradients[:n_dense]
        sparse_grads = gradients[n_dense:]

        # Clip dense gradients (global norm clipping, Sec 4.1.4)
        if self.grad_clip_dense > 0:
            dense_grads_valid = [g for g in dense_grads if g is not None]
            dense_vars_valid = [v for g, v in zip(dense_grads, dense_vars) if g is not None]
            if dense_grads_valid:
                dense_grads_valid, _ = tf.clip_by_global_norm(
                    dense_grads_valid, self.grad_clip_dense
                )
                self.dense_optimizer.apply_gradients(
                    zip(dense_grads_valid, dense_vars_valid)
                )

        # Clip sparse gradients
        if self.grad_clip_sparse > 0:
            sparse_grads_valid = [g for g in sparse_grads if g is not None]
            sparse_vars_valid = [v for g, v in zip(sparse_grads, sparse_vars) if g is not None]
            if sparse_grads_valid:
                sparse_grads_valid, _ = tf.clip_by_global_norm(
                    sparse_grads_valid, self.grad_clip_sparse
                )
                self.sparse_optimizer.apply_gradients(
                    zip(sparse_grads_valid, sparse_vars_valid)
                )

        return total_loss, task_losses


class OneTransTrainer:
    """Training and evaluation manager for OneTrans.

    Implements:
      - Dual-optimizer training (Sec 4.1.4)
      - Next-batch evaluation protocol (Sec 4.1.2)
      - Mixed precision training support (Sec 3.5.2)
      - AUC and UAUC metrics

    Args:
        model: OneTrans model instance.
        config: Training configuration dict.
    """

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}

        # Dual optimizer
        self.optimizer = DualOptimizer(
            model=model,
            dense_lr=self.config.get("dense_lr", 0.005),
            sparse_lr=self.config.get("sparse_lr", 0.01),
            rms_alpha=self.config.get("rms_alpha", 0.99999),
            grad_clip_dense=model.config.grad_clip_dense,
            grad_clip_sparse=model.config.grad_clip_sparse,
        )

        # Mixed precision policy (Sec 3.5.2)
        if self.config.get("mixed_precision", False):
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)

        # Metrics
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.auc_metrics = {}
        for task_name in model.config.task_names:
            self.auc_metrics[task_name] = tf.keras.metrics.AUC(name=f"auc_{task_name}")

    def train_epoch(self, dataset, steps_per_epoch=None, verbose=True):
        """Train for one epoch.

        Args:
            dataset: tf.data.Dataset yielding (inputs, labels) pairs.
            steps_per_epoch: Optional limit on number of steps.
            verbose: Whether to print progress.
        Returns:
            Dict of average metrics.
        """
        self.train_loss_metric.reset_states()
        for metric in self.auc_metrics.values():
            metric.reset_states()

        start_time = time.time()
        step = 0

        for batch in dataset:
            inputs, labels = batch

            total_loss, task_losses = self.optimizer.train_step(inputs, labels)
            self.train_loss_metric.update_state(total_loss)

            step += 1
            if verbose and step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {step}: loss={total_loss:.4f}, "
                      f"time={elapsed:.1f}s")

            if steps_per_epoch and step >= steps_per_epoch:
                break

        metrics = {"loss": self.train_loss_metric.result().numpy()}
        return metrics

    def evaluate(self, dataset, steps=None):
        """Evaluate on a dataset.

        Implements next-batch evaluation: predictions are logged in eval mode,
        then metrics are computed (Sec 4.1.2).

        Args:
            dataset: tf.data.Dataset yielding (inputs, labels) pairs.
            steps: Optional limit on number of eval steps.
        Returns:
            Dict of evaluation metrics (AUC per task).
        """
        for metric in self.auc_metrics.values():
            metric.reset_states()

        step = 0
        for batch in dataset:
            inputs, labels = batch
            predictions = self.model(inputs, training=False)

            for task_name in self.model.config.task_names:
                if task_name in predictions and task_name in labels:
                    self.auc_metrics[task_name].update_state(
                        labels[task_name], predictions[task_name]
                    )

            step += 1
            if steps and step >= steps:
                break

        results = {}
        for task_name, metric in self.auc_metrics.items():
            results[f"auc_{task_name}"] = metric.result().numpy()

        return results

    def next_batch_evaluate(self, dataset, num_steps):
        """Next-batch evaluation protocol (Sec 4.1.2).

        For each mini-batch:
          1. Log predictions in eval mode
          2. Train on the same batch

        AUC and UAUC are computed daily from each day's predictions
        and macro-averaged across days.

        Args:
            dataset: tf.data.Dataset yielding (inputs, labels) pairs.
            num_steps: Number of steps to run.
        Returns:
            Dict of evaluation metrics.
        """
        all_predictions = {task: [] for task in self.model.config.task_names}
        all_labels = {task: [] for task in self.model.config.task_names}

        step = 0
        for batch in dataset:
            inputs, labels = batch

            # Step 1: Eval mode prediction
            predictions = self.model(inputs, training=False)
            for task_name in self.model.config.task_names:
                if task_name in predictions and task_name in labels:
                    all_predictions[task_name].append(predictions[task_name])
                    all_labels[task_name].append(labels[task_name])

            # Step 2: Train on the same batch
            self.optimizer.train_step(inputs, labels)

            step += 1
            if step >= num_steps:
                break

        # Compute AUC over all collected predictions
        results = {}
        for task_name in self.model.config.task_names:
            if all_predictions[task_name]:
                preds = tf.concat(all_predictions[task_name], axis=0)
                labs = tf.concat(all_labels[task_name], axis=0)
                auc_metric = tf.keras.metrics.AUC()
                auc_metric.update_state(labs, preds)
                results[f"auc_{task_name}"] = auc_metric.result().numpy()

        return results

    def train(self, train_dataset, val_dataset=None, epochs=1,
              steps_per_epoch=None, val_steps=None, verbose=True):
        """Full training loop.

        Args:
            train_dataset: Training tf.data.Dataset.
            val_dataset: Optional validation dataset.
            epochs: Number of epochs.
            steps_per_epoch: Steps per epoch.
            val_steps: Validation steps.
            verbose: Whether to print progress.
        Returns:
            Training history dict.
        """
        history = {"train_loss": [], "val_metrics": []}

        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print("-" * 40)

            # Train
            train_metrics = self.train_epoch(
                train_dataset, steps_per_epoch=steps_per_epoch, verbose=verbose
            )
            history["train_loss"].append(train_metrics["loss"])

            if verbose:
                print(f"  Train Loss: {train_metrics['loss']:.4f}")

            # Validate
            if val_dataset is not None:
                val_metrics = self.evaluate(val_dataset, steps=val_steps)
                history["val_metrics"].append(val_metrics)

                if verbose:
                    for k, v in val_metrics.items():
                        print(f"  Val {k}: {v:.4f}")

        return history


def compute_uauc(user_ids, labels, predictions):
    """Compute impression-weighted user-level AUC (UAUC).

    Paper Sec 4.1.2: "Performance is measured by AUC and UAUC
    (impression-weighted user-level AUC)."

    UAUC = sum_u(n_u * AUC_u) / sum_u(n_u)

    where n_u is the number of impressions for user u, and AUC_u
    is the per-user AUC.

    Args:
        user_ids: [N] user ID array.
        labels: [N] binary labels.
        predictions: [N] predicted probabilities.
    Returns:
        uauc: float, impression-weighted UAUC.
    """
    unique_users = np.unique(user_ids)
    total_weight = 0.0
    weighted_auc_sum = 0.0

    for uid in unique_users:
        mask = (user_ids == uid)
        user_labels = labels[mask]
        user_preds = predictions[mask]

        n_u = len(user_labels)

        # Need both positive and negative examples for AUC
        if len(np.unique(user_labels)) < 2:
            continue

        # Compute per-user AUC
        auc_metric = tf.keras.metrics.AUC()
        auc_metric.update_state(user_labels, user_preds)
        user_auc = auc_metric.result().numpy()

        weighted_auc_sum += n_u * user_auc
        total_weight += n_u

    if total_weight == 0:
        return 0.0

    return weighted_auc_sum / total_weight
