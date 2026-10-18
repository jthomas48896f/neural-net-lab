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

## Working agreements
- AGENTS.md in this repo governs how changes are made and proven here.
