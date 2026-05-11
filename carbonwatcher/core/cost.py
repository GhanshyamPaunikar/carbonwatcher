"""Aggregated cost tracking across multiple experiment runs."""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Optional

from carbonwatcher.core.energy import EnergyResult
from carbonwatcher.core.carbon import CarbonResult


@dataclass
class RunRecord:
    run_id: str
    experiment_name: str
    model_name: str
    timestamp: str
    duration_s: float
    energy_kwh: float
    carbon_kg: float
    compute_cost_usd: float
    total_cost_usd: float
    infrastructure: str
    region: str
    tags: List[str]
    metadata: dict
    # Location context
    user_name: str = ""
    organization: str = ""
    user_region: str = ""        # where the user's machine is
    user_region_source: str = "" # how we detected it

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(**d)


class CostTracker:
    """Persists run records to a JSON ledger and computes aggregate stats."""

    def __init__(self, ledger_path: str = "carbonwatcher_ledger.json"):
        self.ledger_path = ledger_path
        self._records: List[RunRecord] = []
        self._load()

    def record_run(
        self,
        run_id: str,
        experiment_name: str,
        model_name: str,
        duration_s: float,
        energy_result: EnergyResult,
        carbon_result: CarbonResult,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        user_name: str = "",
        organization: str = "",
        user_region: str = "",
        user_region_source: str = "",
    ) -> RunRecord:
        record = RunRecord(
            run_id=run_id,
            experiment_name=experiment_name,
            model_name=model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_s=duration_s,
            energy_kwh=energy_result.total_energy_kwh,
            carbon_kg=carbon_result.carbon_kg,
            compute_cost_usd=energy_result.compute_cost_usd,
            total_cost_usd=energy_result.total_cost_usd,
            infrastructure=energy_result.infrastructure,
            region=carbon_result.region,
            tags=tags or [],
            metadata=metadata or {},
            user_name=user_name,
            organization=organization,
            user_region=user_region,
            user_region_source=user_region_source,
        )
        self._records.append(record)
        self._save()
        return record

    def get_all_runs(self) -> List[RunRecord]:
        return list(self._records)

    def get_runs_by_experiment(self, name: str) -> List[RunRecord]:
        return [r for r in self._records if r.experiment_name == name]

    def aggregate(self, runs: Optional[List[RunRecord]] = None) -> dict:
        records = runs if runs is not None else self._records
        if not records:
            return {}
        return {
            "num_runs": len(records),
            "total_duration_h": sum(r.duration_s for r in records) / 3600,
            "total_energy_kwh": sum(r.energy_kwh for r in records),
            "total_carbon_kg": sum(r.carbon_kg for r in records),
            "total_cost_usd": sum(r.total_cost_usd for r in records),
            "avg_energy_kwh": sum(r.energy_kwh for r in records) / len(records),
            "avg_carbon_kg": sum(r.carbon_kg for r in records) / len(records),
            "avg_cost_usd": sum(r.total_cost_usd for r in records) / len(records),
            "most_expensive_run": max(records, key=lambda r: r.total_cost_usd).run_id,
            "highest_carbon_run": max(records, key=lambda r: r.carbon_kg).run_id,
        }

    def efficiency_ranking(self, metric: str = "carbon_per_dollar") -> List[dict]:
        """Rank runs by efficiency. metric: carbon_per_dollar | energy_per_dollar."""
        ranked = []
        for r in self._records:
            if r.total_cost_usd > 0:
                if metric == "carbon_per_dollar":
                    score = r.carbon_kg / r.total_cost_usd
                else:
                    score = r.energy_kwh / r.total_cost_usd
                ranked.append({"run_id": r.run_id, "experiment": r.experiment_name,
                                "model": r.model_name, "score": round(score, 6),
                                "metric": metric})
        return sorted(ranked, key=lambda x: x["score"])

    def _load(self):
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path) as f:
                    data = json.load(f)
                self._records = [RunRecord.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                self._records = []

    def _save(self):
        with open(self.ledger_path, "w") as f:
            json.dump([r.to_dict() for r in self._records], f, indent=2)
