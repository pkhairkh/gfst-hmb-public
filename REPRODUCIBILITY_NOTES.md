# Reproducibility Notes

## What Can Be Done From This Bundle

- Inspect the selected source code and tests.
- Build a reference container from `docker/Dockerfile`.
- Install the package locally with:

```bash
python -m pip install -e '.[dev]'
```

- Run selected local tests that do not depend on omitted fixtures or omitted internal-path defaults.

## What Is Historically Omitted

- Upstream teacher weights and tokenizer assets.
- Raw training checkpoints.
- Raw production report forests.
- Existing configs and remote orchestration helpers with internal workspace assumptions.

## Practical Consequence

This means the public release is not a one-command reproduction of the original heavy runs. It is a source-and-evidence release, not a full operational mirror of the internal training environment.

## Recommended Reading Order

1. `README.md`
2. `paper.md`
3. `RESULTS_SUMMARY.md`
4. `LIMITATIONS_AND_OPEN_QUESTIONS.md`

Users who want to recreate large runs will need to supply their own environment details, teacher asset handling, and safe public configs.
