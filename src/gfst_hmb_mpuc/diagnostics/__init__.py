from .convergence import best_train_ce_drop, classify_signal, summarize_experiment_run, threshold_steps
try:
    from .training_recipe_eval import (
        FastBudgetController,
        build_architecture_diagnosis,
        build_branch_plans,
        build_fast_recommendations,
        build_passive_summary,
        build_recommendations,
        classify_passive_diagnosis,
        collect_source_snapshot,
        rank_branch_results,
        resolve_checkpoint_path,
        run_training_recipe_eval,
    )
except Exception:  # pragma: no cover - optional local dependency surface
    FastBudgetController = None
    build_architecture_diagnosis = None
    build_branch_plans = None
    build_fast_recommendations = None
    build_passive_summary = None
    build_recommendations = None
    classify_passive_diagnosis = None
    collect_source_snapshot = None
    rank_branch_results = None
    resolve_checkpoint_path = None
    run_training_recipe_eval = None

__all__ = [
    "best_train_ce_drop",
    "classify_signal",
    "summarize_experiment_run",
    "threshold_steps",
]

if FastBudgetController is not None:
    __all__.extend(
        [
            "FastBudgetController",
            "build_architecture_diagnosis",
            "build_branch_plans",
            "build_fast_recommendations",
            "build_passive_summary",
            "build_recommendations",
            "classify_passive_diagnosis",
            "collect_source_snapshot",
            "rank_branch_results",
            "resolve_checkpoint_path",
            "run_training_recipe_eval",
        ]
    )

try:
    from .live_stream_eval import run_live_stream_eval
except Exception:  # pragma: no cover - optional local dependency surface
    run_live_stream_eval = None
else:
    __all__.append("run_live_stream_eval")

try:
    from .fixed_windows import (
        build_diagnostic_master_payload,
        load_fixed_window_artifact,
        materialize_diagnostic_subset_payload,
        write_fixed_window_artifact,
    )
except Exception:  # pragma: no cover - optional local dependency surface
    build_diagnostic_master_payload = None
    load_fixed_window_artifact = None
    materialize_diagnostic_subset_payload = None
    write_fixed_window_artifact = None
else:
    __all__.extend(
        [
            "build_diagnostic_master_payload",
            "load_fixed_window_artifact",
            "materialize_diagnostic_subset_payload",
            "write_fixed_window_artifact",
        ]
    )
