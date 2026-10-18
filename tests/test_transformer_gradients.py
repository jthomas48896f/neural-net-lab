"""
Gradient checks for the from scratch transformer.

Every backward method in neural_net_lab/transformer.py was derived by
hand. This suite verifies each one against central finite differences:

    numeric_grad = (f(x + eps) - f(x - eps)) / (2 * eps)

and asserts the relative error

    |analytic - numeric| / max(1e-8, |analytic| + |numeric|)

stays below 1e-5 for every parameter element and every input element,
on small configurations where an exhaustive check is cheap.

Run: python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_net_lab.transformer import (
    LayerNorm, RMSNorm, MultiHeadSelfAttention, FeedForward,
    SwiGLUFeedForward, TransformerBlock,
    CharTransformerLM,
)
from neural_net_lab.losses import CrossEntropyLoss

EPS = 1e-6
TOL = 1e-5
ABS_TOL = 1e-7


def numeric_grad(scalar_fn, arr, eps=EPS):
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = arr[idx]
        arr[idx] = old + eps
        f_plus = scalar_fn()
        arr[idx] = old - eps
        f_minus = scalar_fn()
        arr[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def assert_grad_close(test_case, analytic, numeric, label):
    # Combined criterion: relative error below TOL, or absolute error
    # below ABS_TOL. The absolute floor matters for gradients that are
    # identically zero by symmetry. Example: the key bias gradient in
    # attention is exactly zero, because adding a constant to every key
    # shifts each score row by a constant and softmax is invariant to
    # that. Finite differences return rounding noise (~1e-9) there, and
    # a pure relative test would flag noise as failure.
    denom = np.maximum(1e-8, np.abs(analytic) + np.abs(numeric))
    rel = np.abs(analytic - numeric) / denom
    abs_err = np.abs(analytic - numeric).max()
    test_case.assertTrue(
        rel.max() < TOL or abs_err < ABS_TOL,
        f"{label}: max relative error {rel.max():.3e}, "
        f"max absolute error {abs_err:.3e} (analytic vs numeric mismatch)")


def check_layer(test_case, layer, x, seed=0):
    """Full gradient check of one Layer: input grad + all param grads."""
    rng = np.random.default_rng(seed)
    out = layer.forward(x)
    G = rng.standard_normal(out.shape)

    def scalar_fn():
        return float(np.sum(layer.forward(x) * G))

    for p in layer.parameters():
        p.zero_grad()
    analytic_x = layer.backward(G)
    assert_grad_close(test_case, analytic_x, numeric_grad(scalar_fn, x), "input")

    for i, p in enumerate(layer.parameters()):
        analytic = p.grad.copy()
        numeric = numeric_grad(scalar_fn, p.data)
        assert_grad_close(test_case, analytic, numeric, f"param[{i}] shape {p.data.shape}")


class TestLayerNormGradients(unittest.TestCase):
    def test_gradients(self):
        rng = np.random.default_rng(1)
        layer = LayerNorm(8)
        layer.gamma.data = rng.standard_normal(8)
        layer.beta.data = rng.standard_normal(8)
        x = rng.standard_normal((3, 5, 8))
        check_layer(self, layer, x, seed=1)


# Why: RMSNorm drops LayerNorm's mean-centering term, so its backward is a distinct derivation that needs its own finite-difference proof (input and gain).
class TestRMSNormGradients(unittest.TestCase):
    def test_gradients(self):
        rng = np.random.default_rng(8)
        layer = RMSNorm(8)
        layer.gamma.data = rng.standard_normal(8)
        x = rng.standard_normal((3, 5, 8))
        check_layer(self, layer, x, seed=8)

    def test_gradients_scaled_input(self):
        rng = np.random.default_rng(9)
        layer = RMSNorm(4)
        layer.gamma.data = rng.standard_normal(4)
        x = rng.standard_normal((2, 4, 4)) * 3.0 + 1.5
        check_layer(self, layer, x, seed=9)


# Why: SwiGLU's backward is a product rule across two branches off the same input (the SiLU gate and the up projection), a derivation no other layer here exercises, so it needs its own finite-difference proof.
class TestSwiGLUFeedForwardGradients(unittest.TestCase):
    def test_gradients(self):
        layer = SwiGLUFeedForward(6, d_ff=12, seed=11)
        rng = np.random.default_rng(11)
        x = rng.standard_normal((2, 3, 6))
        check_layer(self, layer, x, seed=11)

    def test_gradients_scaled_input(self):
        layer = SwiGLUFeedForward(4, d_ff=8, seed=12)
        rng = np.random.default_rng(12)
        x = rng.standard_normal((2, 4, 4)) * 3.0 + 1.5
        check_layer(self, layer, x, seed=12)


class TestAttentionGradients(unittest.TestCase):
    def test_causal_gradients(self):
        rng = np.random.default_rng(2)
        layer = MultiHeadSelfAttention(8, 2, causal=True, seed=2)
        x = rng.standard_normal((2, 4, 8))
        check_layer(self, layer, x, seed=2)

    def test_noncausal_gradients(self):
        rng = np.random.default_rng(3)
        layer = MultiHeadSelfAttention(8, 4, causal=False, seed=3)
        x = rng.standard_normal((2, 3, 8))
        check_layer(self, layer, x, seed=3)


class TestFeedForwardGradients(unittest.TestCase):
    def test_gradients(self):
        rng = np.random.default_rng(4)
        layer = FeedForward(8, 16, seed=4)
        x = rng.standard_normal((2, 3, 8))
        check_layer(self, layer, x, seed=4)


class TestTransformerBlockGradients(unittest.TestCase):
    def test_gradients(self):
        rng = np.random.default_rng(5)
        block = TransformerBlock(8, 2, d_ff=16, seed=5)
        x = rng.standard_normal((2, 4, 8))
        check_layer(self, block, x, seed=5)


class TestCharLMGradients(unittest.TestCase):
    def test_gradients(self):
        rng = np.random.default_rng(6)
        model = CharTransformerLM(vocab_size=9, d_model=8, num_heads=2,
                                  num_layers=1, max_seq_len=6, seed=6)
        idx = rng.integers(0, 9, size=(2, 5))
        targets = rng.integers(0, 9, size=(2, 5))
        loss_fn = CrossEntropyLoss()

        def scalar_fn():
            logits = model.forward(idx)
            B, T, V = logits.shape
            return float(loss_fn.forward(logits.reshape(B * T, V),
                                         targets.reshape(B * T)))

        for p in model.parameters():
            p.zero_grad()
        logits = model.forward(idx)
        B, T, V = logits.shape
        loss_fn.forward(logits.reshape(B * T, V), targets.reshape(B * T))
        model.backward(loss_fn.backward().reshape(B, T, V))

        for i, p in enumerate(model.parameters()):
            analytic = p.grad.copy()
            numeric = numeric_grad(scalar_fn, p.data)
            assert_grad_close(self, analytic, numeric,
                              f"lm param[{i}] shape {p.data.shape}")


class TestTrainingSmoke(unittest.TestCase):
    def test_loss_decreases_on_fixed_batch(self):
        from neural_net_lab.optimizers import Adam
        rng = np.random.default_rng(7)
        text = "the quick brown fox jumps over the lazy dog. " * 8
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        data = np.array([stoi[c] for c in text])
        model = CharTransformerLM(vocab_size=len(chars), d_model=32,
                                  num_heads=2, num_layers=2, max_seq_len=16,
                                  seed=7)
        opt = Adam(model.parameters(), lr=0.01)
        loss_fn = CrossEntropyLoss()
        T = 16
        x = data[:T][None, :]
        y = data[1:T + 1][None, :]

        losses = []
        for _ in range(40):
            logits = model.forward(x)
            loss = loss_fn.forward(logits.reshape(-1, len(chars)),
                                   y.reshape(-1))
            losses.append(loss)
            opt.zero_grad()
            model.backward(loss_fn.backward().reshape(1, T, len(chars)))
            opt.step()
        self.assertLess(losses[-1], losses[0],
                        f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}")


if __name__ == "__main__":
    unittest.main()
