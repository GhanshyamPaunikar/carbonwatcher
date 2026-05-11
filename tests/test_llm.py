"""Tests for LLM token-based carbon/cost tracking."""

import pytest
from carbonwatcher.llm.models import lookup_model, list_models, ModelSpec
from carbonwatcher.llm.tracker import LLMTracker, _compute_impact


class TestModelLookup:
    def test_exact_match(self):
        spec = lookup_model("gpt-4o")
        assert spec is not None
        assert spec.provider == "openai"

    def test_prefix_alias(self):
        spec = lookup_model("claude-sonnet-4-6-20250101")
        assert spec is not None
        assert "sonnet" in spec.model_id

    def test_unknown_returns_none(self):
        assert lookup_model("definitely-not-a-model-xyz") is None

    def test_list_all(self):
        models = list_models()
        assert len(models) >= 10

    def test_list_by_provider(self):
        models = list_models(provider="anthropic")
        assert all(m.provider == "anthropic" for m in models)
        assert len(models) >= 3

    def test_mistral_low_carbon_grid(self):
        spec = lookup_model("mistral-large-latest")
        assert spec.provider_grid_intensity_gco2_kwh < 100  # France nuclear grid


class TestComputeImpact:
    def _spec(self) -> ModelSpec:
        return ModelSpec(
            provider="test", model_id="test-model", display_name="Test",
            wh_per_1k_input_tokens=1.0,
            wh_per_1k_output_tokens=4.0,
            provider_grid_intensity_gco2_kwh=500.0,
            cost_per_1m_input_tokens=10.0,
            cost_per_1m_output_tokens=30.0,
        )

    def test_energy_calculation(self):
        result = _compute_impact(self._spec(), input_tokens=1000, output_tokens=1000)
        assert result.input_energy_wh  == pytest.approx(1.0)
        assert result.output_energy_wh == pytest.approx(4.0)
        assert result.total_energy_wh  == pytest.approx(5.0)

    def test_carbon_calculation(self):
        result = _compute_impact(self._spec(), input_tokens=1000, output_tokens=1000)
        # 5 Wh = 0.005 kWh × 500 gCO2/kWh = 2.5 gCO2
        assert result.carbon_g  == pytest.approx(2.5)
        assert result.carbon_kg == pytest.approx(0.0025)

    def test_cost_calculation(self):
        result = _compute_impact(self._spec(), input_tokens=1_000_000, output_tokens=1_000_000)
        assert result.input_cost_usd  == pytest.approx(10.0)
        assert result.output_cost_usd == pytest.approx(30.0)
        assert result.total_cost_usd  == pytest.approx(40.0)

    def test_zero_tokens(self):
        result = _compute_impact(self._spec(), input_tokens=0, output_tokens=0)
        assert result.total_energy_wh == 0.0
        assert result.carbon_g == 0.0
        assert result.total_cost_usd == 0.0

    def test_custom_grid_intensity(self):
        result = _compute_impact(self._spec(), 1000, 1000, grid_intensity=0.0)
        assert result.carbon_g == 0.0


class TestLLMTracker:
    def test_record_and_retrieve(self, tmp_path):
        tracker = LLMTracker(experiment="test", ledger_path=str(tmp_path / "llm.json"))
        result = tracker.record(model="gpt-4o", input_tokens=500, output_tokens=200)
        assert result.provider == "openai"
        assert result.total_tokens == 700

        calls = tracker.get_all_calls()
        assert len(calls) == 1
        assert calls[0].model_id == "gpt-4o"

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "llm.json")
        t1 = LLMTracker(experiment="exp", ledger_path=path)
        t1.record(model="claude-sonnet-4-6", input_tokens=100, output_tokens=50)

        t2 = LLMTracker(experiment="exp", ledger_path=path)
        assert len(t2.get_all_calls()) == 1

    def test_aggregate(self, tmp_path):
        tracker = LLMTracker(ledger_path=str(tmp_path / "llm.json"))
        tracker.record(model="gpt-4o", input_tokens=1000, output_tokens=500)
        tracker.record(model="claude-haiku-4-5-20251001", input_tokens=500, output_tokens=100)

        agg = tracker.aggregate()
        assert agg["num_calls"] == 2
        assert agg["total_input_tokens"] == 1500
        assert agg["total_output_tokens"] == 600

    def test_unknown_model_uses_default(self, tmp_path):
        tracker = LLMTracker(ledger_path=str(tmp_path / "llm.json"))
        # Should not raise
        result = tracker.record(model="some-future-model-xyz", input_tokens=100, output_tokens=50)
        assert result.total_tokens == 150

    def test_compare_models_sorted_by_carbon(self, tmp_path):
        tracker = LLMTracker(ledger_path=str(tmp_path / "llm.json"))
        rows = tracker.compare_models(input_tokens=1000, output_tokens=500)
        carbons = [r["carbon_g"] for r in rows]
        assert carbons == sorted(carbons)

    def test_mistral_lower_carbon_than_gpt4(self, tmp_path):
        tracker = LLMTracker(ledger_path=str(tmp_path / "llm.json"))
        rows = tracker.compare_models(input_tokens=1000, output_tokens=500)
        by_model = {r["model_id"]: r for r in rows}
        assert by_model["mistral-large-latest"]["carbon_g"] < by_model["gpt-4-turbo"]["carbon_g"]

    def test_filter_by_experiment(self, tmp_path):
        path = str(tmp_path / "llm.json")
        # Use a single tracker; experiments are just metadata on records
        tracker = LLMTracker(experiment="expA", ledger_path=path)
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50)

        # Second experiment recorded via a fresh tracker (picks up existing file)
        tracker2 = LLMTracker(experiment="expB", ledger_path=path)
        tracker2.record(model="gpt-4o-mini", input_tokens=200, output_tokens=80)

        # Reload and filter
        t3 = LLMTracker(ledger_path=path)
        assert len(t3.get_calls_by_experiment("expA")) == 1
        assert len(t3.get_calls_by_experiment("expB")) == 1
