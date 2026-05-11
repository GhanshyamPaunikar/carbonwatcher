"""
LLMTracker — records every API call (tokens, energy, carbon, cost)
and persists them to a JSON ledger alongside training runs.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import List, Optional

from carbonwatcher.llm.models import ModelSpec, lookup_model, _ALL_MODELS
from carbonwatcher.core.carbon import _compute_equivalents


@dataclass
class LLMCallResult:
    """Computed impact for a single API call."""
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # Energy
    input_energy_wh: float
    output_energy_wh: float
    total_energy_wh: float
    # Carbon
    carbon_g: float
    carbon_kg: float
    grid_intensity_gco2_kwh: float
    intensity_source: str
    # Cost
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    # Equivalents
    equivalents: dict

    def summary(self) -> str:
        lines = [
            f"Model:    {self.provider} / {self.model_id}",
            f"Tokens:   {self.input_tokens:,} in + {self.output_tokens:,} out = {self.total_tokens:,} total",
            f"Energy:   {self.total_energy_wh*1000:.4f} mWh",
            f"Carbon:   {self.carbon_g:.6f} gCO2e  [grid: {self.grid_intensity_gco2_kwh} gCO2/kWh, {self.intensity_source}]",
            f"Cost:     ${self.total_cost_usd:.6f} USD",
            f"Equiv:    {self.equivalents.get('smartphone_charges', 0):.4f} phone charges",
        ]
        return "\n".join(lines)


@dataclass
class LLMCallRecord:
    """Persisted record of one API call."""
    call_id: str
    experiment: str
    timestamp: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_energy_wh: float
    carbon_g: float
    carbon_kg: float
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    grid_intensity_gco2_kwh: float
    intensity_source: str
    latency_s: float
    tags: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LLMCallRecord":
        return cls(**d)


def _compute_impact(
    spec: ModelSpec,
    input_tokens: int,
    output_tokens: int,
    grid_intensity: Optional[float] = None,
    intensity_source: str = "provider_default",
) -> LLMCallResult:
    intensity = grid_intensity if grid_intensity is not None else spec.provider_grid_intensity_gco2_kwh

    input_wh  = spec.wh_per_1k_input_tokens  * input_tokens  / 1000.0
    output_wh = spec.wh_per_1k_output_tokens * output_tokens / 1000.0
    total_wh  = input_wh + output_wh

    carbon_g  = total_wh / 1000.0 * intensity   # Wh → kWh × gCO2/kWh
    carbon_kg = carbon_g / 1000.0

    input_cost  = spec.cost_per_1m_input_tokens  * input_tokens  / 1_000_000
    output_cost = spec.cost_per_1m_output_tokens * output_tokens / 1_000_000

    return LLMCallResult(
        model_id=spec.model_id,
        provider=spec.provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_energy_wh=input_wh,
        output_energy_wh=output_wh,
        total_energy_wh=total_wh,
        carbon_g=carbon_g,
        carbon_kg=carbon_kg,
        grid_intensity_gco2_kwh=intensity,
        intensity_source=intensity_source,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
        equivalents=_compute_equivalents(carbon_g),
    )


class LLMTracker:
    """
    Records LLM API calls and aggregates token/energy/carbon/cost stats.

    Usage — manual:
        tracker = LLMTracker(experiment="rag-pipeline")
        result = tracker.record(model="claude-sonnet-4-6", input_tokens=500, output_tokens=200)
        print(result.summary())

    Usage — via provider wrappers (automatic):
        from carbonwatcher.llm.anthropic_wrapper import TrackedAnthropic
        client = TrackedAnthropic(experiment="rag-pipeline")
        response = client.messages.create(...)   # tracked automatically
    """

    def __init__(
        self,
        experiment: str = "default",
        ledger_path: str = "carbonwatcher_llm_ledger.json",
        tags: Optional[List[str]] = None,
    ):
        self.experiment = experiment
        self.ledger_path = ledger_path
        self.default_tags = tags or []
        self._records: List[LLMCallRecord] = []
        self._load()

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float = 0.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        custom_grid_intensity: Optional[float] = None,
    ) -> LLMCallResult:
        """
        Record one API call manually.
        model: any model_id from carbonwatcher.llm.models or a prefix alias.
        """
        spec = lookup_model(model)
        if spec is None:
            # Unknown model — use a conservative default
            from carbonwatcher.llm.models import ModelSpec
            spec = ModelSpec(
                provider="unknown", model_id=model, display_name=model,
                wh_per_1k_input_tokens=0.003, wh_per_1k_output_tokens=0.012,
                provider_grid_intensity_gco2_kwh=300.0,
                cost_per_1m_input_tokens=0.0, cost_per_1m_output_tokens=0.0,
            )

        intensity_source = "provider_default"
        if custom_grid_intensity is not None:
            intensity_source = "custom"

        result = _compute_impact(
            spec=spec,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            grid_intensity=custom_grid_intensity,
            intensity_source=intensity_source,
        )

        record = LLMCallRecord(
            call_id=str(uuid.uuid4())[:8],
            experiment=self.experiment,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=spec.model_id,
            provider=spec.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            total_energy_wh=result.total_energy_wh,
            carbon_g=result.carbon_g,
            carbon_kg=result.carbon_kg,
            input_cost_usd=result.input_cost_usd,
            output_cost_usd=result.output_cost_usd,
            total_cost_usd=result.total_cost_usd,
            grid_intensity_gco2_kwh=result.grid_intensity_gco2_kwh,
            intensity_source=intensity_source,
            latency_s=latency_s,
            tags=list(set(self.default_tags + (tags or []))),
            metadata=metadata or {},
        )
        self._records.append(record)
        self._save()
        return result

    def get_all_calls(self) -> List[LLMCallRecord]:
        return list(self._records)

    def get_calls_by_experiment(self, name: str) -> List[LLMCallRecord]:
        return [r for r in self._records if r.experiment == name]

    def aggregate(self, records: Optional[List[LLMCallRecord]] = None) -> dict:
        recs = records if records is not None else self._records
        if not recs:
            return {}
        return {
            "num_calls": len(recs),
            "total_input_tokens": sum(r.input_tokens for r in recs),
            "total_output_tokens": sum(r.output_tokens for r in recs),
            "total_tokens": sum(r.total_tokens for r in recs),
            "total_energy_wh": sum(r.total_energy_wh for r in recs),
            "total_carbon_g": sum(r.carbon_g for r in recs),
            "total_carbon_kg": sum(r.carbon_kg for r in recs),
            "total_cost_usd": sum(r.total_cost_usd for r in recs),
            "avg_cost_per_call": sum(r.total_cost_usd for r in recs) / len(recs),
            "models_used": list({r.model_id for r in recs}),
            "providers_used": list({r.provider for r in recs}),
            "most_used_model": max(
                {r.model_id for r in recs},
                key=lambda m: sum(1 for r in recs if r.model_id == m),
            ),
        }

    def compare_models(
        self,
        input_tokens: int = 1000,
        output_tokens: int = 500,
    ) -> List[dict]:
        """
        Rank all known models by carbon footprint for a fixed token budget.
        Useful for choosing the most efficient model for your use case.
        """
        from carbonwatcher.llm.models import list_models
        rows = []
        for spec in list_models():
            r = _compute_impact(spec, input_tokens, output_tokens)
            rows.append({
                "provider": spec.provider,
                "model": spec.display_name,
                "model_id": spec.model_id,
                "carbon_g": round(r.carbon_g, 6),
                "energy_mwh": round(r.total_energy_wh * 1000, 4),
                "cost_usd": round(r.total_cost_usd, 6),
                "grid_gco2_kwh": spec.provider_grid_intensity_gco2_kwh,
            })
        return sorted(rows, key=lambda x: x["carbon_g"])

    def _load(self):
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path) as f:
                    data = json.load(f)
                self._records = [LLMCallRecord.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError, TypeError):
                self._records = []

    def _save(self):
        with open(self.ledger_path, "w") as f:
            json.dump([r.to_dict() for r in self._records], f, indent=2)
