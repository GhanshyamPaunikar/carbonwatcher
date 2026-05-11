"""Tests for carbon estimation, energy calculation, cost tracking, and parser."""

import json
import pytest
from unittest.mock import patch, MagicMock
from carbonwatcher.core.carbon import CarbonEstimator, _compute_equivalents
from carbonwatcher.core.energy import EnergyCalculator
from carbonwatcher.core.hardware import HardwareProfile
from carbonwatcher.core.cost import CostTracker, RunRecord
from carbonwatcher.core.energy import EnergyResult
from carbonwatcher.core.carbon import CarbonResult


def make_profile(**kwargs) -> HardwareProfile:
    defaults = dict(
        duration_s=3600.0,
        avg_cpu_percent=50.0,
        avg_cpu_power_w=40.0,
        peak_cpu_power_w=60.0,
        avg_dram_power_w=3.0,
        avg_ram_gb=8.0,
        num_gpus=0,
        gpu_names=[],
        avg_gpu_utilization=0.0,
        avg_gpu_power_w=0.0,
        peak_gpu_power_w=0.0,
        total_cpu_energy_kwh=0.04,
        total_gpu_energy_kwh=0.0,
        total_dram_energy_kwh=0.003,
        cpu_model="Test CPU",
        cpu_power_source="tdp_estimate",
    )
    defaults.update(kwargs)
    return HardwareProfile(**defaults)


def make_energy(cost: float = 0.05) -> EnergyResult:
    return EnergyResult(
        raw_energy_kwh=0.01, pue_overhead=1.1, total_energy_kwh=0.011,
        infrastructure="cloud", compute_cost_usd=cost,
    )


def make_carbon(kg: float = 0.00425) -> CarbonResult:
    return CarbonResult(
        energy_kwh=0.011, region="us-east-1",
        grid_intensity_gco2_per_kwh=386.0,
        carbon_kg=kg, carbon_g=kg * 1000, equivalents={},
    )


# ── CarbonEstimator ───────────────────────────────────────────────────────────

class TestCarbonEstimator:
    def test_known_region(self):
        est = CarbonEstimator(region="us-east-1")
        result = est.estimate(1.0)
        assert result.carbon_g == pytest.approx(386.0)
        assert result.carbon_kg == pytest.approx(0.386)

    def test_zero_energy(self):
        est = CarbonEstimator(region="eu-west-1")
        assert est.estimate(0.0).carbon_g == 0.0

    def test_custom_intensity(self):
        est = CarbonEstimator(region="us-east-1", custom_intensity=100.0)
        assert est.estimate(1.0).carbon_g == pytest.approx(100.0)

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError, match="Unknown region"):
            CarbonEstimator(region="mars-north-1")

    def test_equivalents_nonzero(self):
        eq = _compute_equivalents(1000.0)
        assert eq["km_driven"] > 0
        assert eq["smartphone_charges"] > 0

    def test_low_carbon_flag(self):
        assert CarbonEstimator(region="eu-west-1").estimate(1.0).is_low_carbon

    def test_high_carbon_flag(self):
        assert not CarbonEstimator(region="ap-south-1").estimate(1.0).is_low_carbon

    def test_intensity_source_static(self):
        est = CarbonEstimator(region="us-east-1")
        assert est.estimate(1.0).intensity_source == "static_table"

    def test_custom_intensity_source(self):
        est = CarbonEstimator(region="us-east-1", custom_intensity=50.0)
        assert est.estimate(1.0).intensity_source == "custom"


# ── EnergyCalculator ──────────────────────────────────────────────────────────

class TestEnergyCalculator:
    def test_cpu_only_pue(self):
        profile = make_profile(total_cpu_energy_kwh=0.1, total_gpu_energy_kwh=0.0,
                               total_dram_energy_kwh=0.0, num_gpus=0)
        result = EnergyCalculator(infrastructure="cloud").calculate(profile)
        assert result.raw_energy_kwh == pytest.approx(0.1)
        assert result.total_energy_kwh == pytest.approx(0.1 * 1.10)

    def test_dram_included_in_raw(self):
        profile = make_profile(total_cpu_energy_kwh=0.04, total_gpu_energy_kwh=0.0,
                               total_dram_energy_kwh=0.003, num_gpus=0)
        result = EnergyCalculator(infrastructure="cloud").calculate(profile)
        assert result.raw_energy_kwh == pytest.approx(0.043)

    def test_on_premise_pue(self):
        profile = make_profile(total_cpu_energy_kwh=1.0, total_gpu_energy_kwh=0.0,
                               total_dram_energy_kwh=0.0)
        result = EnergyCalculator(infrastructure="on_premise").calculate(profile)
        assert result.total_energy_kwh == pytest.approx(1.5)

    def test_laptop_pue_is_1(self):
        profile = make_profile(total_cpu_energy_kwh=0.5, total_gpu_energy_kwh=0.0,
                               total_dram_energy_kwh=0.0)
        result = EnergyCalculator(infrastructure="laptop").calculate(profile)
        assert result.total_energy_kwh == pytest.approx(0.5)

    def test_unknown_infra_raises(self):
        with pytest.raises(ValueError):
            EnergyCalculator(infrastructure="spaceship")

    def test_custom_pue(self):
        profile = make_profile(total_cpu_energy_kwh=1.0, total_gpu_energy_kwh=0.0,
                               total_dram_energy_kwh=0.0)
        result = EnergyCalculator(infrastructure="cloud", custom_pue=2.0).calculate(profile)
        assert result.total_energy_kwh == pytest.approx(2.0)


# ── CostTracker ───────────────────────────────────────────────────────────────

class TestCostTracker:
    def test_record_and_reload(self, tmp_path):
        ledger = str(tmp_path / "ledger.json")
        tracker = CostTracker(ledger_path=ledger)
        tracker.record_run("r1", "exp-A", "GPT2", 120.0, make_energy(), make_carbon(),
                           tags=["test"])

        loaded = CostTracker(ledger_path=ledger)
        assert len(loaded.get_all_runs()) == 1
        assert loaded.get_all_runs()[0].model_name == "GPT2"

    def test_aggregate_empty(self, tmp_path):
        tracker = CostTracker(ledger_path=str(tmp_path / "empty.json"))
        assert tracker.aggregate() == {}

    def test_efficiency_ranking_order(self, tmp_path):
        ledger = str(tmp_path / "ledger.json")
        tracker = CostTracker(ledger_path=ledger)
        tracker.record_run("r1", "exp", "big",   3600, make_energy(1.0), make_carbon(0.5))
        tracker.record_run("r2", "exp", "small", 3600, make_energy(0.5), make_carbon(0.1))
        ranking = tracker.efficiency_ranking()
        assert ranking[0]["run_id"] == "r2"

    def test_filter_by_experiment(self, tmp_path):
        ledger = str(tmp_path / "ledger.json")
        tracker = CostTracker(ledger_path=ledger)
        tracker.record_run("r1", "exp-A", "m1", 60, make_energy(), make_carbon())
        tracker.record_run("r2", "exp-B", "m2", 60, make_energy(), make_carbon())
        assert len(tracker.get_runs_by_experiment("exp-A")) == 1


# ── HardwareProfile ───────────────────────────────────────────────────────────

class TestHardwareProfile:
    def test_total_energy_includes_dram(self):
        p = make_profile(total_cpu_energy_kwh=0.04,
                         total_gpu_energy_kwh=0.10,
                         total_dram_energy_kwh=0.003)
        assert p.total_energy_kwh == pytest.approx(0.143)

    def test_gpu_label_dedup(self):
        p = make_profile(num_gpus=2, gpu_names=["A100", "A100"])
        assert p.gpu_label == "2× A100"

    def test_gpu_label_none(self):
        p = make_profile(num_gpus=0, gpu_names=[])
        assert "None" in p.gpu_label


# ── Parser ────────────────────────────────────────────────────────────────────

class TestParser:
    def _make_log(self, tmp_path, run_id: str, energy_kwh: float, carbon_g: float,
                  experiment: str = "exp") -> str:
        entry = {
            "run_id": run_id,
            "experiment": experiment,
            "model": "test-model",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "actual": {
                "duration_s": 60.0,
                "energy_kwh": energy_kwh,
                "carbon_g": carbon_g,
                "carbon_kg": carbon_g / 1000,
                "cost_usd": 0.01,
            },
            "location": {"region": "us-east-1", "grid_intensity_gco2_kwh": 386.0,
                         "intensity_source": "static_table"},
            "equivalents": {},
            "epochs": [],
            "tags": [],
            "metadata": {},
        }
        path = str(tmp_path / f"{run_id}.log")
        with open(path, "w") as f:
            json.dump(entry, f)
        return path

    def test_parse_all_logs(self, tmp_path):
        from carbonwatcher import parser
        self._make_log(tmp_path, "r1", 0.01, 3.86)
        self._make_log(tmp_path, "r2", 0.02, 7.72)
        logs = parser.parse_all_logs(str(tmp_path))
        assert len(logs) == 2

    def test_filter_by_experiment(self, tmp_path):
        from carbonwatcher import parser
        self._make_log(tmp_path, "r1", 0.01, 3.86, experiment="train")
        self._make_log(tmp_path, "r2", 0.01, 3.86, experiment="eval")
        logs = parser.parse_all_logs(str(tmp_path), experiment="train")
        assert len(logs) == 1
        assert logs[0]["experiment"] == "train"

    def test_aggregate(self, tmp_path):
        from carbonwatcher import parser
        self._make_log(tmp_path, "r1", 0.01, 3.86)
        self._make_log(tmp_path, "r2", 0.02, 7.72)
        agg = parser.aggregate_logs(parser.parse_all_logs(str(tmp_path)))
        assert agg["num_runs"] == 2
        assert agg["total_energy_kwh"] == pytest.approx(0.03)
        assert agg["total_carbon_g"] == pytest.approx(11.58)

    def test_empty_dir(self, tmp_path):
        from carbonwatcher import parser
        assert parser.parse_all_logs(str(tmp_path)) == []
