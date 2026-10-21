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
    LayerNorm, RMSNorm, RotaryPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    MultiHeadSelfAttention, GroupedQueryAttention, MultiQueryAttention, CrossAttention, FeedForward,
    SwiGLUFeedForward, TransformerBlock, Embedding,
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


# Why: RoPE has no learned parameters and no other layer here encodes position by rotation, so its backward (the transpose rotation) needs its own finite-difference proof of the input gradient, on both sequence and per-head shapes.
class TestRotaryPositionalEmbeddingGradients(unittest.TestCase):
    def test_gradients(self):
        rng = np.random.default_rng(13)
        layer = RotaryPositionalEmbedding(8)
        x = rng.standard_normal((2, 4, 8))
        check_layer(self, layer, x, seed=13)

    def test_gradients_scaled_head_shape(self):
        rng = np.random.default_rng(14)
        layer = RotaryPositionalEmbedding(8)
        x = rng.standard_normal((2, 2, 3, 8)) * 3.0 + 1.5
        check_layer(self, layer, x, seed=14)


# Why: the sinusoidal encoding is a fixed constant with no learned parameters, so its backward is the identity, a claim no learned or rotated positional layer here makes; it needs its own finite-difference proof, plus a direct check of the sin/cos values it adds.
class TestSinusoidalPositionalEmbeddingGradients(unittest.TestCase):
    # Why: proves the identity input gradient against finite differences on a plain sequence shape.
    def test_gradients(self):
        rng = np.random.default_rng(23)
        layer = SinusoidalPositionalEmbedding(8)
        x = rng.standard_normal((2, 4, 8))
        check_layer(self, layer, x, seed=23)

    # Why: proves the same identity gradient holds for a scaled input at a different dimension, where a shape- or scale-dependent error would show.
    def test_gradients_scaled_different_dimension(self):
        rng = np.random.default_rng(24)
        layer = SinusoidalPositionalEmbedding(4)
        x = rng.standard_normal((2, 4, 4)) * 3.0 + 1.5
        check_layer(self, layer, x, seed=24)

    # Why: pins the added values themselves to the sin/cos formula, since a wrong encoding with an identity backward would still pass the gradient checks above.
    def test_encoding_matches_formula(self):
        dim = 8
        layer = SinusoidalPositionalEmbedding(dim)
        out = layer.forward(np.zeros((1, 4, dim)))
        # Position 0: sin(0) = 0 on even features, cos(0) = 1 on odd.
        self.assertTrue(np.allclose(out[0, 0], [0.0, 1.0] * (dim // 2)))
        # A later position, recomputed feature by feature from the formula.
        pos = 3
        expected = np.array([
            np.sin(pos * 10000.0 ** (-2 * (j // 2) / dim)) if j % 2 == 0
            else np.cos(pos * 10000.0 ** (-2 * (j // 2) / dim))
            for j in range(dim)
        ])
        self.assertTrue(np.allclose(out[0, pos], expected))

    # Why: proves position actually enters the output, so identical tokens at different positions do not stay identical.
    def test_identical_tokens_differ_by_position(self):
        layer = SinusoidalPositionalEmbedding(8)
        token = np.arange(8, dtype=float)
        x = np.tile(token, (1, 4, 1))
        out = layer.forward(x)
        self.assertFalse(np.allclose(out[0, 0], out[0, 1]))
        self.assertFalse(np.allclose(out[0, 0], out[0, 3]))


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


# Why: grouped-query attention shares each key/value head across a group of query heads, so its backward adds a group reduction (summing each expanded key/value gradient over its group) that no other attention here exercises; the second test covers the single key/value head limit (multi-query attention), where every query head shares one head.
class TestGroupedQueryAttentionGradients(unittest.TestCase):
    def test_causal_gradients(self):
        rng = np.random.default_rng(15)
        layer = GroupedQueryAttention(8, 4, num_kv_heads=2, causal=True, seed=15)
        x = rng.standard_normal((2, 4, 8))
        check_layer(self, layer, x, seed=15)

    def test_noncausal_single_kv_head_gradients(self):
        rng = np.random.default_rng(16)
        layer = GroupedQueryAttention(8, 4, num_kv_heads=1, causal=False, seed=16)
        x = rng.standard_normal((2, 3, 8))
        check_layer(self, layer, x, seed=16)


# Why: multi-query attention is the standalone single key/value head form: every query head shares one key/value head, so its backward sums the expanded key/value gradients over all query heads, and its key/value projections are sized to a single head; grouped-query attention reaches this only as a limit of its general group machinery, so the standalone block needs its own finite-difference proof.
class TestMultiQueryAttentionGradients(unittest.TestCase):
    def test_causal_gradients(self):
        rng = np.random.default_rng(27)
        layer = MultiQueryAttention(8, 4, causal=True, seed=27)
        x = rng.standard_normal((2, 4, 8))
        check_layer(self, layer, x, seed=27)

    def test_noncausal_gradients_scaled_input(self):
        rng = np.random.default_rng(28)
        layer = MultiQueryAttention(8, 2, causal=False, seed=28)
        x = rng.standard_normal((2, 3, 8)) * 3.0 + 1.5
        check_layer(self, layer, x, seed=28)


# Why: cross attention is the one attention here with a two-input API: queries come from one sequence and keys/values from another, the two sequences may differ in length, and backward must return two input gradients (the query input only through the query projection, the key/value input through the key and value projections summed). No single-input check_layer call can prove that split, so this class checks both inputs and all parameters with its own two-input harness, following the same tolerances.
class TestCrossAttentionGradients(unittest.TestCase):
    def check_cross_attention(self, layer, x_q, x_kv, seed):
        rng = np.random.default_rng(seed)
        out = layer.forward(x_q, x_kv)
        G = rng.standard_normal(out.shape)

        def scalar_fn():
            return float(np.sum(layer.forward(x_q, x_kv) * G))

        for p in layer.parameters():
            p.zero_grad()
        analytic_q, analytic_kv = layer.backward(G)
        assert_grad_close(self, analytic_q, numeric_grad(scalar_fn, x_q), "query input")
        assert_grad_close(self, analytic_kv, numeric_grad(scalar_fn, x_kv), "key/value input")

        for i, p in enumerate(layer.parameters()):
            analytic = p.grad.copy()
            numeric = numeric_grad(scalar_fn, p.data)
            assert_grad_close(self, analytic, numeric, f"param[{i}] shape {p.data.shape}")

    # Why: proves both input gradients and all parameter gradients when the query sequence is shorter than the key/value sequence, the encoder-decoder shape self attention never has.
    def test_gradients_different_lengths(self):
        rng = np.random.default_rng(25)
        layer = CrossAttention(8, 2, seed=25)
        x_q = rng.standard_normal((2, 3, 8))
        x_kv = rng.standard_normal((2, 4, 8))
        self.check_cross_attention(layer, x_q, x_kv, seed=25)

    # Why: proves the same two-input split holds for scaled and shifted inputs at a different head count, where a shape- or scale-dependent error in either branch would show.
    def test_gradients_scaled_inputs(self):
        rng = np.random.default_rng(26)
        layer = CrossAttention(8, 4, seed=26)
        x_q = rng.standard_normal((2, 2, 8)) * 3.0 + 1.5
        x_kv = rng.standard_normal((2, 3, 8)) * 3.0 - 1.0
        self.check_cross_attention(layer, x_q, x_kv, seed=26)


# Why: the embedding lookup's backward is a scatter-add (a repeated index
# accumulates every position's gradient; an unused row gets exactly zero).
# The language model computes it inline, but the standalone layer had no
# check of its own; a backward that overwrote instead of accumulated would
# pass any test that never repeats an index, so both cases are pinned here.
class TestEmbeddingGradients(unittest.TestCase):
    def check_embedding(self, layer, idx, seed):
        rng = np.random.default_rng(seed)
        out = layer.forward(idx)
        G = rng.standard_normal(out.shape)

        def scalar_fn():
            return float(np.sum(layer.forward(idx) * G))

        layer.weight.zero_grad()
        result = layer.backward(G)
        self.assertIsNone(result, "indices are discrete: no input gradient")
        analytic = layer.weight.grad.copy()
        numeric = numeric_grad(scalar_fn, layer.weight.data)
        assert_grad_close(self, analytic, numeric, "embedding weight")
        return analytic

    def test_repeated_indices_gradients(self):
        layer = Embedding(7, 5, seed=21)
        # Row 3 appears three times, row 0 twice; rows 1 and 5 never appear.
        idx = np.array([[3, 0, 3], [6, 3, 0]])
        analytic = self.check_embedding(layer, idx, seed=21)
        self.assertTrue(np.all(analytic[1] == 0), "unused row 1 gradient is exactly zero")
        self.assertTrue(np.all(analytic[5] == 0), "unused row 5 gradient is exactly zero")

    def test_single_position_gradients(self):
        layer = Embedding(6, 4, seed=22)
        idx = np.array([[2], [4]])
        self.check_embedding(layer, idx, seed=22)


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
