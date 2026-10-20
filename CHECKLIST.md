# The 110 percent checklist: Machine

Ascent alignment: Layer 5, Autonomy. Aligned October 15, 2026 (see "The need" in the README and NEEDS.md in the profile repo).
State as of October 15, 2026. Every line carries its evidence; a line without evidence does not count as done. This file is maintained by the evolution pass.

## 100 percent: completeness
- [x] Every file of the verified local publish tree present on GitHub, proven by tree diff at launch (October 14 to 15, 2026)
- [x] README accurate against the code, including the published results: 7 of 7 gradient checks, char LM validation loss 5.297 to 2.413 over 250 steps, PyTorch benchmark 4.13x faster per step at identical parameters
- [x] LICENSE present
- [x] COLLABORATE.md present
- [x] CONTRIBUTING.md present
- [x] CI workflow present (tests plus the short training run on every push)

## The extra 10 percent
- [x] CI green on GitHub runners (verified October 15, 2026)
- [x] Repo description and topics set (description states the verified results)
- [x] Good first issues open: #2 run the PyTorch benchmark on your machine, #4 add a gradient check for the optimizer step (both labeled good first issue, sized in hours; verified open 2026-10-17, and #1 closed by PR #3)
- [x] Tagged release: v0.1.0, first verified baseline (published, marked Latest)
- [x] Live demo: https://topsoilcommons.github.io/machine/ (render verified October 15, 2026)
- [x] Fresh baseline rerun October 15, 2026: python3 -m unittest discover -s tests, 7 tests, OK
- [x] First outside contribution merged October 16, 2026: pull request #3 by shyam1017, finite difference gradient checks for the Dense, BatchNorm, and Dropout layers (good first issue #1, now closed). Verified before merging by running the full suite on the PR branch: 16 tests, OK (7 baseline plus 9 new). Suite on main after the merge: 16 tests, OK. The original 7 of 7 transformer gradient checks in the line above are unchanged; the suite total is now 16.
- [x] Scale step training run October 16, 2026: 628,932 parameters (d_model 128, 3 layers, 4 heads, context 128), validation loss 5.396 to 2.441 over 200 steps on the sonnets corpus, wall time 852 s, artifact results/char_lm_run_scale2.json. Steps were set to 200 rather than the baseline's 250 because profiling at this size projected a 250 step run at or over the roughly 20 minute wall time budget for the pass; the final validation loss is slightly above the baseline's 2.413, and the README says so.

- [x] SwiGLU gated feed-forward added October 18, 2026: SwiGLUFeedForward in neural_net_lab/transformer.py (SiLU gate times an up projection, then a down projection), exported from the package, with finite difference gradient checks in tests/test_transformer_gradients.py (TestSwiGLUFeedForwardGradients: standard and scaled input). Suite 18 tests OK before, 20 tests OK after.
- [x] Rotary positional embedding added October 18, 2026: RotaryPositionalEmbedding in neural_net_lab/transformer.py (rotate-half RoPE, no learned parameters, backward is the transpose rotation), exported from the package, with finite difference gradient checks in tests/test_transformer_gradients.py (TestRotaryPositionalEmbeddingGradients: standard sequence shape and scaled per-head shape). Suite 20 tests OK before, 22 tests OK after.
- [x] Grouped-query attention added October 19, 2026: GroupedQueryAttention in neural_net_lab/transformer.py (num_heads query heads share num_kv_heads key/value heads; the backward sums each expanded key/value gradient over its group, a reduction no other layer here exercises), exported from the package, with finite difference gradient checks in tests/test_transformer_gradients.py (TestGroupedQueryAttentionGradients: causal with 2 of 4 key/value heads, and non-causal with a single key/value head, the multi-query limit; max absolute error 4.6e-9 against central finite differences across both configurations). Suite 22 tests OK before, 24 tests OK after.
- [x] Standalone embedding layer added October 19, 2026 (evening): Embedding in neural_net_lab/transformer.py (gather forward, scatter-add backward with np.add.at, the derivation the character language model computes inline, now verified on its own), exported from the package, with finite difference gradient checks in tests/test_transformer_gradients.py (TestEmbeddingGradients: repeated indices, where one row accumulates three positions of gradient and two unused rows are exactly zero, and a single position per index shape; max absolute error 3.1e-10 against central finite differences across both configurations). Suite 24 tests OK before, 26 tests OK after.
- [x] Sinusoidal positional embedding added October 20, 2026: SinusoidalPositionalEmbedding in neural_net_lab/transformer.py (fixed sin/cos encodings precomputed at construction, no learned parameters, backward is the identity for the input gradient), exported from the package, with finite difference gradient checks in tests/test_transformer_gradients.py (TestSinusoidalPositionalEmbeddingGradients: standard sequence shape and scaled input at a different dimension, encoding values at position 0 and position 3 matched against the sin/cos formula, and identical tokens at different positions produce different outputs; max absolute error 7.9e-9 against central finite differences across both gradient configurations). Suite 26 tests OK before, 30 tests OK after.

- [x] Cross attention added October 20, 2026 (evening): CrossAttention in neural_net_lab/transformer.py (queries from one sequence, keys and values from another, query and key/value lengths may differ, no causal mask; the one two-input API here, forward(x_q, x_kv) and backward returning (grad_x_q, grad_x_kv), where the key/value input gradient sums the key and value projection branches), exported from the package, with finite difference gradient checks in tests/test_transformer_gradients.py (TestCrossAttentionGradients: different query and key/value lengths, and scaled and shifted inputs at a different head count, both input gradients and all parameter gradients checked with a two-input harness; max absolute error 7.1e-8 against central finite differences across both configurations). Suite 30 tests OK before, 32 tests OK after.

## Working agreements
- AGENTS.md in this repo governs how changes are made and proven here.
