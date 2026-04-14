from __future__ import annotations

import argparse
import signal
from dataclasses import replace
from pathlib import Path

from .diagnostics.fixed_windows import materialize_diagnostic_subset_payload as _materialize_diagnostic_subset_payload
from .training.config import TrainingArtifactsConfig, TrainingModelConfig, TrainingRunConfig, load_training_run_config
from .training.engine import (
    _apply_training_loss,
    _bootstrap_fingerprint,
    _bootstrap_manifest,
    _bootstrap_passes,
    _build_optimizer,
    _default_runtime_state,
    _distill_kl_per_token,
    _maybe_promote_ramp,
    _phase_index,
    _schedule_scale,
    _signal_flag_summary,
    _summarise_group_diagnostics,
    run_distillation,
)


def _install_graceful_signal_handlers() -> dict[int, signal.Handlers]:
    previous: dict[int, signal.Handlers] = {}

    def _handle_signal(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_signal)
    return previous


def _restore_signal_handlers(previous: dict[int, signal.Handlers]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YAML-driven distillation entrypoint for Qwen3 teacher -> MPUC GFST-HMB student"
    )
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warm-start-from", default=None, help="load student weights only from a validated checkpoint")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _apply_overrides(config: TrainingRunConfig, *, output_dir: str | None, device: str | None) -> TrainingRunConfig:
    model = config.model
    artifacts = config.artifacts
    if device is not None:
        model = replace(model, device=str(device))
    if output_dir is not None:
        artifacts = replace(artifacts, output_dir=str(output_dir))
    if model is config.model and artifacts is config.artifacts:
        return config
    return replace(config, model=model, artifacts=artifacts)


def main() -> None:
    args = _parse_args()
    config = load_training_run_config(Path(args.config))
    effective_config = _apply_overrides(config, output_dir=args.output_dir, device=args.device)
    previous_handlers = _install_graceful_signal_handlers()
    try:
        run_distillation(
            effective_config,
            resume=bool(args.resume),
            warm_start_from=None if args.warm_start_from is None else str(args.warm_start_from),
        )
    except KeyboardInterrupt:
        return
    finally:
        _restore_signal_handlers(previous_handlers)


if __name__ == "__main__":
    main()
