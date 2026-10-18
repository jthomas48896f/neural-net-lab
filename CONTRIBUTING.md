# Contributing to Machine

Contributions are welcome, and the door is kept visibly open: there
is always at least one good first issue available. The house rule is
the evidence rule: a contribution ships with the run that proves it
(the gradient check output, the benchmark artifact), not just the
code.

## Setup

```bash
pip install numpy matplotlib
python -m unittest discover -s tests   # gradient checks + training smoke
python main.py                         # the original demos
```

The benchmark script also needs PyTorch, which the lab itself does
not depend on. Install it (CPU wheel) before running the benchmark:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Good first issues (sized in hours)

- **Add a check for the optimizer step (1 to 2 hours, issue #4).**
  The gradient checks cover the transformer and the original layers
  (Dense, BatchNorm, and Dropout, added in PR #3). The optimizers in
  `neural_net_lab/optimizers.py` have no equivalent check. Pick one
  optimizer (SGD with momentum or Adam), drive it on a small
  quadratic problem where the correct step is known, and add a test
  in the existing style that shows the parameters move by the
  expected amount.
- **Grow the eval corpus (about an hour).** Add one public domain
  text to `data/` with its source named, and record the trained
  model's loss on it next to the sonnets run. Comparisons across
  corpora teach more than single numbers.
- **Run the benchmark on your machine (under an hour, issue #2).**
  Install PyTorch first (see Setup above), then run
  `python scripts/benchmark_pytorch.py --out /tmp/benchmark_pytorch.json`
  and file the resulting JSON in issue #2. Write to a new file: the
  default `--out` overwrites the published
  `results/benchmark_pytorch.json`. The JSON records the PyTorch
  thread count, but not your CPU model, so note the CPU model in
  the issue yourself (for example from `lscpu` or your system
  information). The published table is one machine; more machines
  make it a measurement instead of an anecdote.
- **Write the missing explanation (1 to 2 hours).** The README
  explains that the key bias gradient is zero by symmetry. Pick one
  other gradient in `transformer.py` and write its derivation as a
  comment or a short docs note, checked against the finite
  difference test.

## Ground rules

- No framework code in the core library. PyTorch appears only in
  the benchmark script, as the measured opponent.
- Every new backward pass lands with its gradient check in the same
  change. An unverified backward does not merge.
- Numbers in the README come only from artifacts in `results/`.
  If you change a number, the artifact that produced it changes in
  the same commit.

## Pull requests

Keep them small and single purpose. State the evidence in the PR
description: what you ran and what it showed. Contributors are
named in release notes.

## Ways in, by level

**First hour.** Run the suite: python -m unittest discover -s tests. It stands at 18 tests, all green. Then run the PyTorch benchmark (scripts/benchmark_pytorch.py) on your machine and file the result: the published table is one machine, and more machines make it a measurement instead of an anecdote.

**First week.** Add a gradient check for one optimizer step (SGD with momentum, or Adam) on a small problem where the correct step is known, in the style of the existing gradient tests. It stands open, labeled good first issue. The transformer and layer checks at a tolerance of 1e-5 are the pattern to match.

**The deep end.** The from scratch library runs about 4.13 times slower per step than PyTorch at identical size. Closing that gap in pure Python, while the gradient checks and the full suite stay green, is this project's hardest honest problem. The benchmark script is the judge.
