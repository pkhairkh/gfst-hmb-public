# Project Structure

This staged GitHub bundle includes:

- `src/gfst_hmb_mpuc/`: selected Python package code for model state, runtime logic, diagnostics, training support modules, and CUDA/C++ extension sources.
- `tests/`: selected unit tests and diagnostics tests.
- `paper.md`: mathematical and architectural reference text.
- Release notes at the repository root.

Intentionally omitted from this staged bundle:

- `assets/upstream/`: upstream teacher/tokenizer assets.
- `reports/`: raw production artifact tree.
- `configs/`: current configs contain machine-specific paths and were not rewritten.
- `docker/`: the working container recipe embedded internal workspace assumptions and was not rewritten.
- Internal remote orchestration scripts with hardcoded host, container, or workspace assumptions.
- Generated metadata such as `*.egg-info`, caches, and binary artifacts.

The result is a public-facing subset rather than a byte-for-byte mirror of the working repository.
