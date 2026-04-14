# GFST-HMB Public Source Release

GFST-HMB is a research codebase for a long-context memory architecture built around a custom MPUC-first student, Grassmann-flag state tracking, and block-based exact-token memory with sparse associative readout. In this repository, the student is not presented as a standard KV-cache transformer or a Mamba runtime. The included code focuses on the GFST-HMB runtime, diagnostics, CUDA extensions, and test coverage that were used to support the project’s current engineering evidence.

This public bundle is a staged release prepared from a larger working repository. It contains selected source, tests, and release documentation. It does not include raw checkpoints, upstream teacher weights, local caches, or remote-operations files that would expose private infrastructure assumptions.

This public release is licensed under the Apache License 2.0. See LICENSE.

This is a code-first public release. It is not a complete artifact mirror of the internal working repository.

## Architecture At A Glance

The project combines three main pieces:

- An MPUC-first student backbone for streaming state updates.
- A GFST local state path that tracks compact geometric summaries.
- A block memory and sparse readout path aimed at exact-token recovery without standard transformer KV caching.

The mathematical reference remains [`paper.md`](./paper.md). The code and evidence in this release should be read as research-engineering work in progress rather than as a final validation package.

## Current Evidence Status

The strongest public evidence in this release is systems and diagnostics evidence:

- A recorded canonical training run at sequence length 256 shows validation CE improving from `12.2674` at step `99` to `10.9624` at step `299`, with teacher gap improving from `3.2528` to `1.9479`.
- In that same run, top-1 block recall at sequence length 256 was `1.0` across the recorded evaluations in `eval.jsonl`.
- Exact-token recovery remained limited in the same evaluation stream, ranging from `0.0391` to `0.1230` exact token recall at length 256.
- Parity, handoff, boundary-transition, and canonical-summary probes exist and support the claim that substantial runtime verification work was done.

The evidence does not justify framing the project as a fully closed exact-memory result.

## What Is In This Public Release

- Selected `src/` package contents.
- Selected `tests/`.
- `pyproject.toml`.
- `paper.md`.
- Release notes and results summaries under the repository root.

## What Is Not In This Public Release

- Raw checkpoints such as `.pt`, `.pth`, or `.ckpt` files.
- Upstream teacher/tokenizer weights.
- Local caches, build outputs, and transient artifacts.
- Internal remote orchestration scripts and configs with baked-in private or machine-specific paths.
- Full raw report trees from the working repository.

## License Status

This public release is licensed under the Apache License 2.0. See LICENSE.

## Reproducibility Scope

This release is suitable for source inspection, limited local testing, and review of the project’s engineering direction. Historically heavy runs were executed in a remote containerized workflow and depended on omitted upstream model assets and internal path conventions. Public users should not assume that the largest training or runtime probes can be reproduced immediately from this bundle alone.

See [`PROJECT_STRUCTURE.md`](./PROJECT_STRUCTURE.md), [`RESULTS_SUMMARY.md`](./RESULTS_SUMMARY.md), [`REPRODUCIBILITY_NOTES.md`](./REPRODUCIBILITY_NOTES.md), and [`LIMITATIONS_AND_OPEN_QUESTIONS.md`](./LIMITATIONS_AND_OPEN_QUESTIONS.md) for release boundaries.
