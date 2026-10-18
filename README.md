<style>
.commons-bar { padding: 1em 0; }
.commons-bar a, .commons-bar span { margin-right: 1em; }
.commons-bar span[aria-current="page"] { font-weight: bold; }
.commons-footer { margin-top: 3em; }
</style>

<nav class="commons-bar" aria-label="Topsoil Commons sites">
  <a href="https://topsoilcommons.github.io/topsoilcommons/"><svg class="bar-mark" viewBox="0 0 24 24" width="17" height="17" aria-hidden="true" focusable="false" style="flex:none;align-self:center;margin-right:7px"><g fill="currentColor"><rect x="4" y="2.5" width="16" height="2"/><rect x="4" y="5.5" width="16" height="2"/><rect x="4" y="8.5" width="16" height="2"/><rect x="4" y="11.5" width="4.5" height="8"/><rect x="15.5" y="11.5" width="4.5" height="8"/><rect x="2" y="20.5" width="20" height="2.2"/></g></svg>Topsoil Commons</a>
  <a href="https://topsoilcommons.github.io/topsoilcommons/foundations.html">Foundations</a>
  <a href="https://topsoilcommons.github.io/ground/">Ground</a>
  <a href="https://topsoilcommons.github.io/still/">Still</a>
  <a href="https://topsoilcommons.github.io/atlas/">Atlas</a>
  <a href="https://topsoilcommons.github.io/cohort/">Cohort</a>
  <a href="https://topsoilcommons.github.io/graph/">Graph</a>
  <span aria-current="page">Machine</span>
  <a href="https://topsoilcommons.github.io/lumen/">Lumen</a>
</nav>

# Machine

Machine teaches how a neural network works by building one from scratch in pure Python and NumPy. Its transformer passes gradient checks 7 of 7 against finite differences, and the full suite stands at 22 tests, all green.

Layer 5 of Topsoil Commons: autonomy.

First step: read the sonnets result in The proof below, where a 358,052 parameter transformer brought validation loss from 5.297 to 2.413 in 250 steps, then run the gradient test: python -m unittest discover -s tests.

Here you learn how a neural network works by building one from scratch in NumPy. You get code
you can read, run, and change. The library has a custom Tensor that
works out gradients, layers, and tools that update the model during
training. It also has a decoder-only transformer. Tests check its
gradients against finite differences. The core library does not call
a framework. A framework shows up only in one benchmark script,
where the script measures the library against it.

## What it is

- **Custom Tensor** with computational graph and reverse-mode
  automatic differentiation (`neural_net_lab/tensor.py`)
- **Layers**: Dense, Dropout, BatchNorm, and the transformer stack
  below
- **Transformer** (`neural_net_lab/transformer.py`): layer
  normalization, RMS normalization (RMSNorm, verified by the
  finite difference gradient checks in
  `tests/test_transformer_gradients.py`), rotary positional
  embedding (RotaryPositionalEmbedding, RoPE, verified by the same
  gradient checks), multi-head causal self
  attention, feed-forward
  blocks (including a SwiGLU gated variant, verified by the same
  gradient checks), pre-norm residual blocks, token and positional embeddings,
  and a character level language model. Every backward pass is hand
  derived in the same explicit style as the original layers.
- **Activations**: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU
- **Losses**: MSE, Cross-Entropy, Binary Cross-Entropy
- **Optimizers**: SGD (momentum/Nesterov), Adam, RMSProp, AdaGrad,
  with step decay and cosine annealing schedules
- **Toy datasets and visualizations**: spirals, moons, decision
  boundaries, training curves, gradient flow

## The proof

Every number below comes from a run executed in this repository and
recorded as a JSON artifact in `results/`. If a number is not in an
artifact, it is not claimed here.

**Gradient checks** (`python -m unittest discover -s tests`, run
2026-10-15, all passing): every transformer backward (layer norm,
rotary positional embedding, causal and non-causal attention, feed-forward, a full block, and the
complete language model) matches central finite differences to a
relative tolerance of 1e-5, with an absolute floor for gradients
that are identically zero by symmetry. The key bias gradient is the
teaching example: it is exactly zero, because adding a constant to
every key shifts each attention score row by a constant and softmax
is invariant to that. The suite also includes a training smoke test:
loss decreases on a fixed batch.

**Character level training run** (`scripts/train_char_lm.py`,
`results/char_lm_run.json`): a 358,052 parameter transformer
(d_model 96, 3 layers, 4 heads, context 96) trained on Shakespeare's
sonnets (96,427 characters, vocabulary 68, public domain, Project
Gutenberg 1041) for 250 steps with Adam.

- Validation loss: **5.297 at step 0 to 2.413 at step 250**
  (training loss 5.256 to 2.377)
- Mean step time 1.76 s in that environment (NumPy 1.26.4, a small
  shared CPU). Wall time 508 s.
- The generated sample in the artifact is, honestly, sonnet-shaped
  static: line lengths and punctuation are learned, the words are
  not. That is what 250 steps buys at this scale, and the artifact
  shows it rather than a cherry-picked later run.

**Sampling quality, measured** (`scripts/sample_quality.py`,
`results/sampling_quality.json`, run 2026-10-17): one small model
(112,964 parameters, d_model 64, 2 layers, trained 120 steps, loss
5.300 to 2.484) sampled 300 characters from the same prompt at
three temperatures, each sample measured the same way. At
temperature 0.5 the sample collapses toward repetition: its most
common character fills 22.7 percent of the sample and only 31.4
percent of its character pairs are distinct, against 59.5 percent
distinct pairs in a same-length window of the real corpus. At
temperature 1.0 the pair figure rises to 50.5 percent. At
temperature 1.5 it reads 72.9 percent, above the corpus window,
and that is not quality: at that temperature the sampling is
closer to random, and random text also scores high on distinct
pairs. The honest reading is the one the numbers support together:
low temperature repeats, high temperature wanders, and neither is
writing. One model, one seed, one prompt; the artifact says so.

**Scale step: character level training run at the next size**
(`scripts/train_char_lm.py`, `results/char_lm_run_scale2.json`,
run 2026-10-16): a 628,932 parameter transformer (d_model 128,
3 layers, 4 heads, context 128, batch 16) trained on the same
Shakespeare's sonnets corpus (96,427 characters, vocabulary 68)
with the same seed and Adam settings. That is about 1.76x the
parameter count of the 358,052 parameter baseline above. A short
profiling run at this configuration projected a 250 step run at
or over the roughly 20 minute wall time budget for this pass, so
this run uses 200 steps rather than a smaller model; the tradeoff
is recorded here instead of hidden.

- Validation loss: **5.396 at step 0 to 2.441 at step 200**
  (training loss 5.379 to 2.403)
- Mean step time 3.70 s in that environment (NumPy 1.26.4, a small
  shared CPU). Wall time 852 s.
- Honest comparison: the final validation loss, 2.441 after
  200 steps, is slightly above the smaller baseline's 2.413 after
  250 steps, so the larger model did not beat the baseline loss
  in this run; it reached a comparable loss with 50 fewer steps.
- The generated sample in the artifact is, honestly, still
  sonnet-shaped static: line lengths, punctuation, and some short
  real words are learned, the words mostly are not, and the
  sentences do not hold together. The scale step alone did not
  fix sample quality at this step count, and the artifact shows
  it rather than a cherry-picked later run.

**Second benchmark task: character level training run on a second
corpus** (`scripts/train_char_lm.py --data
data/alice_in_wonderland.txt`, `results/alice_char_lm_run.json`):
the same configuration and step count as the sonnets run, trained
on Alice's Adventures in Wonderland by Lewis Carroll (145,275
characters, vocabulary 80, public domain, Project Gutenberg 11,
first published 1865) for 250 steps with Adam. The parameter count
differs from the sonnets run because the vocabulary is larger, and
both counts are printed by the code, not asserted by hand.

- Parameters: **360,368** (counted by the code)
- Validation loss: **5.725 at step 0 to 2.417 at step 250**
  (training loss 5.673 to 2.414)
- Mean step time 1.64 s in that environment (NumPy 1.26.4, a small
  shared CPU). Wall time 477 s.
- This was a single run, recorded as measured. The generated sample
  in the artifact has recognizable words and dialogue punctuation,
  but its sentences do not hold together. That is the same honest
  limit as the sonnets sample at this scale, on a prose corpus
  instead of verse, which is what a second task is for: the loss
  curves of the two corpora can now be compared in `results/`.

**PyTorch benchmark** (`scripts/benchmark_pytorch.py`,
`results/benchmark_pytorch.json`): an equivalent PyTorch transformer
(same configuration, identical parameter count of 358,052, same
pre-generated batches, same Adam settings, defaults for threading
on both sides, run 2026-10-15):

| | NumPy lab | PyTorch 2.5.1+cpu |
| --- | --- | --- |
| Mean training step | 498.5 ms | 120.6 ms |
| Eval loss, before 30 steps | 5.256 | 4.356 |
| Eval loss, after 30 steps | 2.867 | 2.529 |

PyTorch is about **4.1x faster per step** at this size, and the two
models reach comparable loss after the same steps on the same data.
The gap is real and smaller than folklore suggests, because at this
scale kernel launch and Python overhead dominate both sides.
Initial losses differ because the random initializations differ;
the step time and the after-steps loss on shared batches are the
comparison. The benchmark is not part of the library's dependencies;
it is a measuring instrument, run once and recorded.

### Quick start

```bash
pip install numpy matplotlib
python main.py                          # the original 10 demos, figures to figures/
python -m unittest discover -s tests    # gradient checks + training smoke
python scripts/train_char_lm.py         # the 250 step sonnet run (~9 minutes)
python scripts/train_char_lm.py --steps 200 --d-model 128 --n-layer 3 --n-head 4 --block-size 128 --batch-size 16 --out results/char_lm_run_scale2.json   # the 200 step scale run (~14 minutes)
python scripts/train_char_lm.py --data data/alice_in_wonderland.txt --corpus "Alice's Adventures in Wonderland by Lewis Carroll" --source "Project Gutenberg ebook 11, public domain" --prompt "Alice was beginning" --out results/alice_char_lm_run.json   # the 250 step Alice run (~8 minutes)
```

### Project structure

```
neural_net_lab/
  tensor.py        - Custom tensor with autograd
  layers.py        - Dense, Dropout, BatchNorm, Parameter
  transformer.py   - LayerNorm, RMSNorm, RoPE, attention, blocks (incl. SwiGLU), char LM
  activations.py   - Activation functions
  losses.py        - Loss functions
  optimizers.py    - Optimizers and LR schedulers
  network.py       - Sequential model builder and training loop
  datasets.py      - Procedural toy dataset generators
  visualization.py - Plotting utilities
scripts/
  train_char_lm.py       - Character LM training runs (writes results/)
  benchmark_pytorch.py   - NumPy vs PyTorch comparison (needs torch)
tests/
  test_layer_gradients.py
  test_transformer_gradients.py
data/
  shakespeare_sonnets.txt - Public domain corpus (Gutenberg 1041)
  alice_in_wonderland.txt - Public domain corpus (Gutenberg 11)
results/
  char_lm_run.json, char_lm_run_scale2.json,
  alice_char_lm_run.json, benchmark_pytorch.json
```

### Status and limitations

- The transformer uses explicit forward/backward methods (the lab's
  Layer pattern), not the Tensor autograd, so the two halves of the
  lab teach the same mathematics in two styles. Unifying them is a
  known next step, not a hidden gap.
- CPU only, and slow by design: this is a teaching and measurement
  instrument, not a training framework. Anyone who needs speed
  should use PyTorch, and the benchmark above says by how much, at
  this scale, on this machine.
- The language model is character level and tiny. Its sample output
  is included in the run artifact precisely so its limits are
  visible.

### Requirements

- Python 3.8+
- NumPy, Matplotlib (PyTorch only for the optional benchmark script)

## The need

This project serves Layer 5: Autonomy. "The machinery of the age opened up, so that understanding is not the property of a priesthood. A society that cannot explain its own machines will be governed by whoever can." Machine meets that need in a plain way. It builds machine learning from scratch in NumPy. Tests check the transformer gradients against finite differences. So a reader can see each step and explain how the machine learns.

Unmet, it looks like: whole populations subject to systems they cannot question, skills and standing allocated by black boxes, dependence dressed up as convenience.

The full map: [The Ascent, a map of human needs](https://github.com/topsoilcommons/topsoilcommons/blob/main/NEEDS.md)

> **Machine, Layer 5: autonomy**, in the Ascent, built on one
> thesis: technology should increase human capability without taking
> human agency away.
>
> - Ground (the body): [ground](https://topsoilcommons.github.io/ground/)
> - Still (the mind at peace): [still](https://topsoilcommons.github.io/still/)
> - Atlas (learning): [atlas](https://github.com/topsoilcommons/atlas)
> - Cohort (belonging): [cohort](https://github.com/topsoilcommons/cohort)
> - Graph (shared truth): [graph](https://github.com/topsoilcommons/graph)
> - Machine (autonomy, Layer 5): this repo
> - Lumen (trust and meaning): [lumen](https://github.com/topsoilcommons/lumen)
>
> This layer's problem family is opaque machine intelligence. Its
> north star metric is students who can explain how the machine
> learns. No number is claimed for it yet; the lab is the instrument,
> and the count starts when teaching with it starts.
>
> The frame is the post AI renaissance, stated as aspiration and
> direction of travel, never as achievement: intelligence understood
> from first principles, in the open, with measurements.

## Live demo

[Live demo](https://topsoilcommons.github.io/machine/)

## Contribute, human or agent

Pick up a good first issue in this repo: each one is sized in hours and written to be finishable by a newcomer.
Take the Eval Challenge against this repo's published benchmark and gradient check results: beat a baseline, or document a real failure, with the run attached.
Report adoption from real use: if you taught with this lab, learned from it, or built on it, tell us what happened, good or bad.

The full protocol: [The Open Table](https://github.com/topsoilcommons/topsoilcommons/blob/main/OPEN_TABLE.md). Verified work is credited in the public [Ledger](https://github.com/topsoilcommons/topsoilcommons/blob/main/LEDGER.md).

### Start with a prompt

For an AI agent:

```
You are an AI agent contributing to Machine. Run the Eval Challenge end to end.
Install with pip install numpy matplotlib, then run python -m unittest discover -s tests -v and python scripts/train_char_lm.py --steps 30 --eval-every 30 --out /tmp/ci_run.json, and run python scripts/benchmark_pytorch.py if you use the benchmark.
Compare your run with the published baselines: gradient checks pass at a relative tolerance of 1e-5, the character level run reaches validation loss 5.297 at step 0 to 2.413 at step 250, and PyTorch is about 4.1x faster per step.
Beat a baseline or document a real failure.
Attach the run, keep to one improvement per pull request, and name your operator.
```

For a human:

```
File a failure report in this repo and include the input you used.
State what happened and what should have happened instead.
Attach the run or the output you saw.
```

<p>The layer below: <a href="https://topsoilcommons.github.io/graph/">Graph, shared truth</a></p>
<p>The layer above: <a href="https://topsoilcommons.github.io/lumen/">Lumen, trust and meaning</a></p>
<p>The research under this layer: <a href="https://topsoilcommons.github.io/topsoilcommons/foundations.html#artificial-intelligence-and-human-computer-interaction">Foundations, Artificial intelligence and human computer interaction</a></p>

Built on Graph: shared truth.

Unlocks the ability to build.

<footer class="commons-footer">
  <p>Topsoil Commons addresses the hierarchy of human needs through open source.</p>
  <p>Motto: Stand on the shoulders of giants.</p>
  <p>One rule: raise the floor beneath everyone.</p>
  <p>Its record is filed in <a href="https://topsoilcommons.github.io/index/">the Index</a>.</p>
  <p>Advised by Groundskeeper.</p>
  <p><a href="https://www.linkedin.com/company/topsoil-commons/">Topsoil Commons on LinkedIn</a></p>
</footer>
