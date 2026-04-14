from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_launcher_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "launch_stagea_write_gate_recovery.py"
    spec = importlib.util.spec_from_file_location("launch_stagea_write_gate_recovery", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage1_continue_rule_requires_two_hits():
    launcher = _load_launcher_module()
    baseline = {"step": 0, "write_usage_kl_to_uniform": 1.0}
    current = {
        "step": 24,
        "write_active_cells_per_block_at_0_10": 1.0,
        "write_second_mass": 0.098,
        "write_top1_mass": 0.184,
        "write_top4_mass": 0.470,
        "write_usage_kl_to_uniform": 0.95,
    }
    conditions = launcher._stage1_continue_conditions(
        baseline_row=baseline,
        current_train_row=current,
        current_eval_row=None,
    )
    assert conditions["write_second_mass_ge_0p0975"] is True
    assert conditions["write_top1_lt_0p185"] is True
    assert launcher._stage1_should_continue(conditions) is True


def test_stage1_continue_rule_stops_with_only_one_hit():
    launcher = _load_launcher_module()
    baseline = {"step": 0, "write_usage_kl_to_uniform": 1.0}
    current = {
        "step": 24,
        "write_active_cells_per_block_at_0_10": 1.0,
        "write_second_mass": 0.09375,
        "write_top1_mass": 0.1875,
        "write_top4_mass": 0.46875,
        "write_usage_kl_to_uniform": 0.89,
    }
    conditions = launcher._stage1_continue_conditions(
        baseline_row=baseline,
        current_train_row=current,
        current_eval_row=None,
    )
    assert conditions["write_usage_kl_improved_10pct"] is True
    assert sum(1 for value in conditions.values() if value) == 1
    assert launcher._stage1_should_continue(conditions) is False


def test_stage_success_requires_write_side_band_and_usage_shift():
    launcher = _load_launcher_module()
    baseline = {"step": 0, "write_usage_kl_to_uniform": 1.0}
    final_eval = {
        "step": 99,
        "write_active_cells_per_block_at_0_10": 2.5,
        "write_top1_mass": 0.179,
        "write_second_mass": 0.102,
        "write_top4_mass": 0.51,
        "read_selected_cells_per_head_mean_focus": 2.4,
        "write_usage_kl_to_uniform": 0.70,
    }
    conditions = launcher._stage_success_conditions(
        baseline_row=baseline,
        final_train_row=None,
        final_eval_row=final_eval,
    )
    assert conditions == {
        "write_active_in_band": True,
        "write_top1_le_0p18": True,
        "write_second_mass_ge_0p10": True,
        "write_top4_ge_0p50": True,
        "read_focus_in_band": True,
        "write_usage_kl_improved_25pct": True,
    }
    assert launcher._stage_is_successful(conditions) is True


def test_build_stage_config_uses_paper_faithful_sparse_defaults():
    launcher = _load_launcher_module()
    config = launcher._build_stage_config(
        source_config={},
        run_dir=Path("/tmp/demo"),
        run_id="demo",
        steps=100,
        train_length=1024,
        eval_length=1024,
        checkpoint_every=25,
        recipe=launcher.STAGE1_RECOVER_RECIPE,
    )
    assert config["model"]["centroid_assignment_top_k"] == 2
    assert config["model"]["centroid_train_support_k"] == 2
    assert config["model"]["centroid_min_selected_cells_per_head"] == 2
    assert config["model"]["centroid_block_support_threshold"] == 0.10
    assert config["model"]["centroid_target_active_cells_per_block"] == 3.5
    assert config["centroid_replay"]["pointer_weight"] == 0.05
    assert config["centroid_replay"]["write_recovery_mode"] == "recover"
    assert config["phases"]["phase_centroid_write_usage_balance_weight"] == [2.0, 2.0, 2.0]
