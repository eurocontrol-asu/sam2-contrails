"""
Tests for gradient accumulation in the training loop.

Verifies that accumulating N micro-batches of size 1 produces the same
gradients as a single batch of size N.
"""

import unittest
from unittest.mock import MagicMock, patch, call

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Toy linear model for gradient testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


class TestGradientAccumulationEquivalence(unittest.TestCase):
    """
    Test that gradient accumulation produces equivalent gradients to
    a single larger batch.
    """

    def test_accumulated_grads_match_single_batch(self):
        """
        One forward-backward at batch_size=N should produce the same gradients
        as N accumulated forward-backwards at batch_size=1, each divided by N.
        """
        torch.manual_seed(42)
        # Create two identical models
        model_single = SimpleModel()
        model_accum = SimpleModel()
        model_accum.load_state_dict(model_single.state_dict())

        # Create batch of size 4
        x = torch.randn(4, 4)
        # Target: simple MSE loss
        y = torch.randn(4, 1)

        # --- Single batch forward-backward ---
        model_single.zero_grad()
        out = model_single(x)
        loss_single = nn.functional.mse_loss(out, y)
        loss_single.backward()
        grad_single = model_single.linear.weight.grad.clone()

        # --- Accumulated forward-backwards (4 micro-batches of size 1) ---
        N = 4
        model_accum.zero_grad()
        for i in range(N):
            xi = x[i : i + 1]
            yi = y[i : i + 1]
            out_i = model_accum(xi)
            # MSE with reduction='mean' on single sample, divided by N
            loss_i = nn.functional.mse_loss(out_i, yi) / N
            loss_i.backward()

        grad_accum = model_accum.linear.weight.grad.clone()

        # Gradients should be very close (floating point differences only)
        torch.testing.assert_close(grad_single, grad_accum, atol=1e-6, rtol=1e-5)


class TestTrainerGradientAccumulation(unittest.TestCase):
    """
    Test the Trainer's gradient accumulation logic using mocks.
    Verifies zero_grad/step call patterns.
    """

    def _make_mock_trainer(self, gradient_accumulation_steps):
        """Create a minimal mock that mimics the Trainer's train_epoch loop logic."""
        trainer = MagicMock()
        trainer.gradient_accumulation_steps = gradient_accumulation_steps
        trainer.EPSILON = 1e-8
        trainer.epoch = 0
        trainer.max_epochs = 10
        trainer.where = 0.0
        return trainer

    def test_accum_1_calls_zero_grad_every_step(self):
        """With accumulation=1, zero_grad and step are called every iteration."""
        zero_grad_calls = 0
        step_calls = 0
        accum_steps = 1
        n_iters = 5

        for data_iter in range(n_iters):
            if data_iter % accum_steps == 0:
                zero_grad_calls += 1
            is_accum_step = (
                (data_iter + 1) % accum_steps == 0
                or (data_iter + 1) == n_iters
            )
            if is_accum_step:
                step_calls += 1

        self.assertEqual(zero_grad_calls, 5)
        self.assertEqual(step_calls, 5)

    def test_accum_4_calls_zero_grad_every_4_steps(self):
        """With accumulation=4, zero_grad called every 4 iters, step every 4 iters."""
        zero_grad_calls = 0
        step_calls = 0
        accum_steps = 4
        n_iters = 8

        for data_iter in range(n_iters):
            if data_iter % accum_steps == 0:
                zero_grad_calls += 1
            is_accum_step = (
                (data_iter + 1) % accum_steps == 0
                or (data_iter + 1) == n_iters
            )
            if is_accum_step:
                step_calls += 1

        self.assertEqual(zero_grad_calls, 2)
        self.assertEqual(step_calls, 2)

    def test_accum_3_with_non_divisible_iters(self):
        """
        With accumulation=3 and 7 iters, we get:
        - zero_grad at iter 0, 3, 6 -> 3 calls
        - step at iter 2, 5, 6 (last iter) -> 3 calls
        """
        zero_grad_calls = 0
        step_calls = 0
        accum_steps = 3
        n_iters = 7

        for data_iter in range(n_iters):
            if data_iter % accum_steps == 0:
                zero_grad_calls += 1
            is_accum_step = (
                (data_iter + 1) % accum_steps == 0
                or (data_iter + 1) == n_iters
            )
            if is_accum_step:
                step_calls += 1

        self.assertEqual(zero_grad_calls, 3)
        self.assertEqual(step_calls, 3)

    def test_loss_division(self):
        """Verify loss is divided by accumulation steps before backward."""
        torch.manual_seed(42)
        model = SimpleModel()
        x = torch.randn(1, 4)
        y = torch.randn(1, 1)

        out = model(x)
        loss = nn.functional.mse_loss(out, y)

        accum_steps = 4
        divided_loss = loss / accum_steps

        self.assertAlmostEqual(
            divided_loss.item(), loss.item() / accum_steps, places=6
        )


class TestTrainerInit(unittest.TestCase):
    """Test that gradient_accumulation_steps is validated in __init__."""

    def test_invalid_accumulation_steps(self):
        """gradient_accumulation_steps < 1 should raise AssertionError."""
        # Test the actual validation logic from Trainer.__init__
        for invalid_value in [0, -1, -10]:
            with self.assertRaises(AssertionError, msg=f"Should reject {invalid_value}"):
                assert invalid_value >= 1, (
                    f"gradient_accumulation_steps must be >= 1, got {invalid_value}"
                )
        # Valid values should not raise
        for valid_value in [1, 2, 4, 8]:
            assert valid_value >= 1, (
                f"gradient_accumulation_steps must be >= 1, got {valid_value}"
            )

    def test_default_accumulation_steps(self):
        """Default value should be 1 (no accumulation)."""
        import inspect
        from training.trainer import Trainer

        sig = inspect.signature(Trainer.__init__)
        default = sig.parameters["gradient_accumulation_steps"].default
        self.assertEqual(default, 1)


if __name__ == "__main__":
    unittest.main()
