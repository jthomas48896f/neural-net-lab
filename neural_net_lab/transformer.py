"""
Transformer components built from scratch on NumPy.

Implements a decoder-only transformer block (multi-head causal self
attention, feed-forward network, layer normalization, residuals) and a
small character level language model, following the same Layer pattern
as layers.py: explicit forward methods with hand derived backward
methods, verified against numerical gradients in
tests/test_transformer_gradients.py.

Nothing here calls an external framework. The point of the lab is that
every gradient below was derived by hand and then checked by finite
differences.
"""

import numpy as np

from .layers import Layer, Parameter, Dense
from .activations import Softmax


class LayerNorm(Layer):
    """
    Layer normalization over the last axis.

    y = gamma * (x - mean) / sqrt(var + eps) + beta

    Uses the biased variance (divide by N), matching the standard
    formulation used in transformer blocks.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = Parameter(np.ones(num_features))
        self.beta = Parameter(np.zeros(num_features))

    def forward(self, x):
        self._x = x
        self._mean = x.mean(axis=-1, keepdims=True)
        self._var = x.var(axis=-1, keepdims=True)
        self._std_inv = 1.0 / np.sqrt(self._var + self.eps)
        self._x_hat = (x - self._mean) * self._std_inv
        return self.gamma.data * self._x_hat + self.beta.data

    def backward(self, grad_output):
        D = self.num_features
        # Parameter gradients: sum over all leading axes
        lead_axes = tuple(range(grad_output.ndim - 1))
        self.gamma.grad += (grad_output * self._x_hat).sum(axis=lead_axes)
        self.beta.grad += grad_output.sum(axis=lead_axes)

        dx_hat = grad_output * self.gamma.data
        # Standard layer norm backward (per row of the last axis)
        dx = (self._std_inv / D) * (
            D * dx_hat
            - dx_hat.sum(axis=-1, keepdims=True)
            - self._x_hat * (dx_hat * self._x_hat).sum(axis=-1, keepdims=True)
        )
        return dx

    def parameters(self):
        return [self.gamma, self.beta]


class RMSNorm(Layer):
    """
    Root mean square normalization over the last axis.

    y = gamma * x / sqrt(mean(x^2) + eps)

    Unlike LayerNorm there is no mean centering and no bias: the
    scale is set by the root mean square alone, and the only learned
    parameter is the gain gamma.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = Parameter(np.ones(num_features))

    def forward(self, x):
        self._x = x
        self._rms_inv = 1.0 / np.sqrt(
            np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        self._x_hat = x * self._rms_inv
        return self.gamma.data * self._x_hat

    def backward(self, grad_output):
        # Parameter gradient: sum over all leading axes
        lead_axes = tuple(range(grad_output.ndim - 1))
        self.gamma.grad += (grad_output * self._x_hat).sum(axis=lead_axes)

        dx_hat = grad_output * self.gamma.data
        # RMSNorm backward (per row of the last axis): like LayerNorm
        # without the mean(dx_hat) term, since there is no centering.
        # dx = rms_inv * (dx_hat - x_hat * mean(dx_hat * x_hat))
        dx = self._rms_inv * (
            dx_hat
            - self._x_hat * (dx_hat * self._x_hat).mean(axis=-1, keepdims=True)
        )
        return dx

    def parameters(self):
        return [self.gamma]


class MultiHeadSelfAttention(Layer):
    """
    Multi-head self attention with an optional causal mask.

    Input shape: (B, T, D). Projections q, k, v are computed with three
    learned matrices, split into heads, and combined with a learned
    output projection. In causal mode, position t may only attend to
    positions <= t, which is what makes the block usable for language
    modeling.
    """

    def __init__(self, d_model, num_heads, causal=True, seed=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)
        self.W_q = Parameter(rng.standard_normal((d_model, d_model)) * scale)
        self.W_k = Parameter(rng.standard_normal((d_model, d_model)) * scale)
        self.W_v = Parameter(rng.standard_normal((d_model, d_model)) * scale)
        self.W_o = Parameter(rng.standard_normal((d_model, d_model)) * scale)
        self.b_q = Parameter(np.zeros(d_model))
        self.b_k = Parameter(np.zeros(d_model))
        self.b_v = Parameter(np.zeros(d_model))
        self.b_o = Parameter(np.zeros(d_model))
        self._softmax = Softmax()

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, _, T, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)

    def forward(self, x):
        B, T, D = x.shape
        self._x = x
        q = x @ self.W_q.data + self.b_q.data
        k = x @ self.W_k.data + self.b_k.data
        v = x @ self.W_v.data + self.b_v.data
        self._q = self._split_heads(q)
        self._k = self._split_heads(k)
        self._v = self._split_heads(v)

        scores = self._q @ self._k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        if self.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
        self._attn = self._softmax.forward(scores)
        context = self._attn @ self._v            # (B, H, T, Dh)
        merged = self._merge_heads(context)        # (B, T, D)
        self._merged = merged
        return merged @ self.W_o.data + self.b_o.data

    def backward(self, grad_output):
        B, T, D = grad_output.shape
        # Output projection
        self.W_o.grad += self._merged.reshape(-1, D).T @ grad_output.reshape(-1, D)
        self.b_o.grad += grad_output.sum(axis=(0, 1))
        d_merged = grad_output @ self.W_o.data.T
        d_context = d_merged.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # context = attn @ v
        d_attn = d_context @ self._v.transpose(0, 1, 3, 2)
        d_v = self._attn.transpose(0, 1, 3, 2) @ d_context

        # Softmax backward (masked positions have attn ~ 0, so their
        # score gradients come out ~ 0 as well)
        d_scores = self._softmax.backward(d_attn) / np.sqrt(self.head_dim)

        # scores = q @ k^T / sqrt(Dh)
        d_q = d_scores @ self._k
        d_k = d_scores.transpose(0, 1, 3, 2) @ self._q

        dq = self._merge_heads(d_q)
        dk = self._merge_heads(d_k)
        dv = self._merge_heads(d_v)

        x2 = self._x.reshape(-1, D)
        self.W_q.grad += x2.T @ dq.reshape(-1, D)
        self.b_q.grad += dq.sum(axis=(0, 1))
        self.W_k.grad += x2.T @ dk.reshape(-1, D)
        self.b_k.grad += dk.sum(axis=(0, 1))
        self.W_v.grad += x2.T @ dv.reshape(-1, D)
        self.b_v.grad += dv.sum(axis=(0, 1))

        return dq @ self.W_q.data.T + dk @ self.W_k.data.T + dv @ self.W_v.data.T

    def parameters(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o,
                self.b_q, self.b_k, self.b_v, self.b_o]


class FeedForward(Layer):
    """Position wise feed-forward network: Linear -> GELU -> Linear."""

    def __init__(self, d_model, d_ff=None, seed=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        rng = np.random.default_rng(seed)
        # Dense uses the global RNG for init; reseed around construction
        # so a model's parameters are reproducible from one seed.
        state = np.random.get_state()
        np.random.seed(rng.integers(0, 2 ** 31 - 1))
        self.fc1 = Dense(d_model, d_ff, activation="gelu", weight_init="he")
        self.fc2 = Dense(d_ff, d_model, activation=None, weight_init="xavier")
        np.random.set_state(state)

    def forward(self, x):
        shape = x.shape
        h = self.fc1.forward(x.reshape(-1, shape[-1]))
        out = self.fc2.forward(h)
        return out.reshape(shape)

    def backward(self, grad_output):
        shape = grad_output.shape
        g = self.fc2.backward(grad_output.reshape(-1, shape[-1]))
        g = self.fc1.backward(g)
        return g.reshape(shape)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()


class SwiGLUFeedForward(Layer):
    """
    Gated feed-forward network (SwiGLU): Linear -> SiLU gate times a
    second Linear projection -> Linear.

    h = SiLU(x @ W_gate + b_gate) * (x @ W_up + b_up)
    y = h @ W_down + b_down

    The gate makes the backward a product rule across two branches off
    the same input, which is why this variant carries its own gradient
    check rather than reusing the FeedForward one.
    """

    def __init__(self, d_model, d_ff=None, seed=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        rng = np.random.default_rng(seed)
        # Dense uses the global RNG for init; reseed around construction
        # so a model's parameters are reproducible from one seed.
        state = np.random.get_state()
        np.random.seed(rng.integers(0, 2 ** 31 - 1))
        self.gate = Dense(d_model, d_ff, activation=None, weight_init="he")
        self.up = Dense(d_model, d_ff, activation=None, weight_init="he")
        self.down = Dense(d_ff, d_model, activation=None, weight_init="xavier")
        np.random.set_state(state)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        self._gate_out = self.gate.forward(x2)
        self._up_out = self.up.forward(x2)
        self._sigmoid = 1.0 / (1.0 + np.exp(-self._gate_out))
        h = self._gate_out * self._sigmoid * self._up_out
        out = self.down.forward(h)
        return out.reshape(shape)

    def backward(self, grad_output):
        shape = grad_output.shape
        dh = self.down.backward(grad_output.reshape(-1, shape[-1]))
        # Product rule: h = silu(g) * u, silu'(g) = s * (1 + g * (1 - s))
        d_up = dh * (self._gate_out * self._sigmoid)
        d_gate = dh * self._up_out * self._sigmoid * (
            1.0 + self._gate_out * (1.0 - self._sigmoid))
        grad = self.up.backward(d_up) + self.gate.backward(d_gate)
        return grad.reshape(shape)

    def parameters(self):
        return (self.gate.parameters() + self.up.parameters()
                + self.down.parameters())


class TransformerBlock(Layer):
    """
    Pre-norm transformer block.

    x = x + Attention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
    """

    def __init__(self, d_model, num_heads, d_ff=None, causal=True, seed=None):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, causal=causal, seed=rng.integers(0, 2 ** 31 - 1))
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, seed=rng.integers(0, 2 ** 31 - 1))

    def forward(self, x):
        self._res1 = x + self.attn.forward(self.ln1.forward(x))
        return self._res1 + self.ffn.forward(self.ln2.forward(self._res1))

    def backward(self, grad_output):
        # Second residual: grad flows to res1 directly and through the ffn branch
        d_res1 = grad_output + self.ln2.backward(self.ffn.backward(grad_output))
        # First residual: grad flows to x directly and through the attention branch
        return d_res1 + self.ln1.backward(self.attn.backward(d_res1))

    def parameters(self):
        return (self.ln1.parameters() + self.attn.parameters()
                + self.ln2.parameters() + self.ffn.parameters())


class CharTransformerLM:
    """
    A small decoder-only character level language model.

    Token embedding + learned positional embedding, a stack of
    TransformerBlocks, a final layer norm, and a linear head back to the
    vocabulary. Loss is the lab's CrossEntropyLoss over all positions.
    """

    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4,
                 max_seq_len=128, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tok_embedding = Parameter(
            rng.standard_normal((vocab_size, d_model)) * np.sqrt(2.0 / d_model))
        self.tok_embedding.grad = np.zeros_like(self.tok_embedding.data)
        self.pos_embedding = Parameter(
            rng.standard_normal((max_seq_len, d_model)) * 0.1)
        self.pos_embedding.grad = np.zeros_like(self.pos_embedding.data)
        self.blocks = [
            TransformerBlock(d_model, num_heads,
                             seed=rng.integers(0, 2 ** 31 - 1))
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(d_model)
        self.head_W = Parameter(
            rng.standard_normal((d_model, vocab_size)) * np.sqrt(2.0 / d_model))
        self.head_W.grad = np.zeros_like(self.head_W.data)
        self.head_b = Parameter(np.zeros(vocab_size))
        self.head_b.grad = np.zeros_like(self.head_b.data)

    def forward(self, idx):
        """idx: integer array (B, T). Returns logits (B, T, V)."""
        B, T = idx.shape
        self._idx = idx
        x = self.tok_embedding.data[idx] + self.pos_embedding.data[:T][None, :, :]
        for block in self.blocks:
            x = block.forward(x)
        self._final = self.ln_f.forward(x)
        return self._final @ self.head_W.data + self.head_b.data

    def backward(self, grad_logits):
        """grad_logits: (B, T, V). Fills .grad on every parameter."""
        B, T, V = grad_logits.shape
        D = self.d_model
        self.head_W.grad += self._final.reshape(-1, D).T @ grad_logits.reshape(-1, V)
        self.head_b.grad += grad_logits.sum(axis=(0, 1))
        g = grad_logits @ self.head_W.data.T
        g = self.ln_f.backward(g)
        for block in reversed(self.blocks):
            g = block.backward(g)
        # Embedding gradients
        np.add.at(self.tok_embedding.grad, self._idx, g)
        self.pos_embedding.grad[:T] += g.sum(axis=0)
        return g

    def parameters(self):
        params = [self.tok_embedding, self.pos_embedding]
        for block in self.blocks:
            params += block.parameters()
        params += self.ln_f.parameters()
        params += [self.head_W, self.head_b]
        return params

    def count_parameters(self):
        return int(sum(p.data.size for p in self.parameters()))

    def generate(self, idx, max_new_tokens, temperature=1.0, rng=None):
        """Autoregressive sampling. idx: 1D array of starting token ids."""
        rng = rng or np.random.default_rng()
        idx = np.array(idx, dtype=int).reshape(1, -1)
        for _ in range(max_new_tokens):
            window = idx[:, -self.max_seq_len:]
            logits = self.forward(window)
            last = logits[0, -1] / max(temperature, 1e-8)
            probs = np.exp(last - last.max())
            probs /= probs.sum()
            nxt = rng.choice(self.vocab_size, p=probs)
            idx = np.concatenate([idx, np.array([[nxt]])], axis=1)
        return idx[0]
