"""CarbonTracker — context manager, decorator, and epoch-based API for ML experiments.

Epoch-based usage (recommended for training loops):
    tracker = CarbonTracker(epochs=100)
    for epoch in range(100):
        tracker.epoch_start()
        train(...)
        tracker.epoch_end()     # logs per-epoch stats + prints prediction
    tracker.stop()

Context manager usage (single block):
    with CarbonTracker() as ct:
        train(...)
    print(ct.result.summary())

Decorator usage:
    @CarbonTracker.track(experiment="exp", model="ResNet50")
    def train(): ...
"""

import uuid
import time
import functools
from dataclasses import dataclass
from typing import Callable, List, Optional

from carbonwatcher.core.hardware import HardwareProfiler, HardwareProfile
from carbonwatcher.core.energy import EnergyCalculator, EnergyResult
from carbonwatcher.core.carbon import CarbonEstimator, CarbonResult
from carbonwatcher.core.cost import CostTracker, RunRecord


@dataclass
class EpochResult:
    epoch: int
    hardware: HardwareProfile
    energy: EnergyResult
    carbon: CarbonResult

    def summary(self) -> str:
        return (
            f"  Epoch {self.epoch}  |  "
            f"time: {self.hardware.duration_s:.1f}s  |  "
            f"energy: {self.energy.total_energy_kwh*1000:.4f} Wh  |  "
            f"carbon: {self.carbon.carbon_g:.6f} gCO2e  |  "
            f"CPU src: {self.hardware.cpu_power_source}"
        )


@dataclass
class TrackingResult:
    run_id: str
    experiment_name: str
    model_name: str
    hardware: HardwareProfile
    energy: EnergyResult
    carbon: CarbonResult
    record: RunRecord
    epoch_results: List[EpochResult]

    def summary(self) -> str:
        lines = [
            f"Run:        {self.run_id}",
            f"Experiment: {self.experiment_name}  |  Model: {self.model_name}",
            f"CPU:        {self.hardware.cpu_model or 'Unknown'}"
            f"  [{self.hardware.cpu_power_source}]",
            f"GPU:        {self.hardware.gpu_label}",
            f"Duration:   {self.hardware.duration_s:.1f}s",
            f"Energy:     {self.energy.total_energy_kwh*1000:.4f} Wh"
            f"  (CPU: {self.hardware.total_cpu_energy_kwh*1000:.4f}"
            f"  GPU: {self.hardware.total_gpu_energy_kwh*1000:.4f}"
            f"  DRAM: {self.hardware.total_dram_energy_kwh*1000:.4f})  PUE×{self.energy.pue_overhead:.2f}",
            f"Carbon:     {self.carbon.carbon_g:.6f} gCO2e"
            f"  ({self.carbon.carbon_kg:.8f} kg)",
            f"Grid:       {self.carbon.region}"
            f" @ {self.carbon.grid_intensity_gco2_per_kwh} gCO2/kWh"
            f"  [{self.carbon.intensity_source}]",
            f"Cost:       ${self.record.total_cost_usd:.4f} USD",
            f"Equiv:      {self.carbon.equivalents['smartphone_charges']:.4f} phone charges"
            f"  |  {self.carbon.equivalents['km_driven']:.6f} km driven",
        ]
        if self.epoch_results:
            lines.append(f"Epochs:     {len(self.epoch_results)}")
        return "\n".join(lines)


class CarbonTracker:
    """
    Track energy, carbon, and cost of ML training runs.

    Parameters
    ----------
    epochs           : Total epochs planned (enables prediction after monitor_epochs).
    monitor_epochs   : How many epochs to monitor before extrapolating (default: 1).
    experiment_name  : Label for grouping runs in the ledger.
    model_name       : Model identifier (free text).
    region           : Cloud region or country for carbon intensity lookup.
    infrastructure   : "cloud" | "on_premise" | "laptop" | "workstation".
    num_vcpus        : vCPUs allocated (used for CPU cost when no GPU).
    gpu_model        : GPU model name for cost lookup when NVML unavailable.
    storage_gb       : Persistent storage allocated (for storage cost).
    sample_interval_s: Hardware sampling frequency (seconds).
    ledger_path      : JSON file to persist run records.
    log_dir          : Directory for structured log files (None = no logs).
    tags             : Arbitrary string tags stored with each run.
    metadata         : Arbitrary dict stored with each run.
    custom_pue       : Override PUE (Power Usage Effectiveness).
    custom_intensity : Override grid carbon intensity (gCO2/kWh).
    custom_gpu_price : Override GPU price (USD/GPU-hour).
    verbose          : 0 = silent, 1 = per-epoch summary + final, 2 = debug.
    ignore_errors    : If True, measurement errors don't raise exceptions.
    """

    def __init__(
        self,
        epochs: int = 1,
        monitor_epochs: int = 1,
        experiment_name: str = "default",
        model_name: str = "unknown",
        region: Optional[str] = None,       # None = auto-detect from location
        infrastructure: str = "cloud",
        num_vcpus: int = 8,
        gpu_model: str = "default",
        storage_gb: float = 0.0,
        sample_interval_s: float = 1.0,
        ledger_path: str = "carbonwatcher_ledger.json",
        log_dir: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        custom_pue: Optional[float] = None,
        custom_intensity: Optional[float] = None,
        custom_gpu_price: Optional[float] = None,
        verbose: int = 1,
        ignore_errors: bool = False,
    ):
        self.epochs = max(epochs, 1)
        self.monitor_epochs = max(monitor_epochs, 1)
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.verbose = verbose
        self.ignore_errors = ignore_errors
        self.log_dir = log_dir
        self.tags = tags or []
        self.metadata = metadata or {}

        # Auto-detect user location if no region specified
        from carbonwatcher.core.location import get_user_region, get_config_value
        self._user_region, self._user_region_source = get_user_region(region)
        self._user_name = get_config_value("user_name", "")
        self._organization = get_config_value("organization", "")

        if verbose >= 1 and self._user_region_source != "config" and region is None:
            print(f"[carbonwatcher] Location: {self._user_region} "
                  f"(detected via {self._user_region_source}). "
                  f"Set permanently: carbonwatcher config --region <region>")

        self._profiler = HardwareProfiler(sample_interval_s=sample_interval_s)
        self._energy_calc = EnergyCalculator(
            infrastructure=infrastructure,
            num_vcpus=num_vcpus,
            gpu_model=gpu_model,
            storage_gb=storage_gb,
            custom_pue=custom_pue,
            custom_gpu_price=custom_gpu_price,
        )
        self._carbon_est = CarbonEstimator(
            region=self._user_region,
            custom_intensity=custom_intensity,
        )
        self._cost_tracker = CostTracker(ledger_path=ledger_path)

        self._run_id: str = ""
        self._current_epoch: int = 0
        self._epoch_results: List[EpochResult] = []
        self.result: Optional[TrackingResult] = None

        # Accumulated totals across epochs for prediction
        self._monitored_energy_kwh: float = 0.0
        self._monitored_duration_s: float = 0.0

    # ── epoch-based API ───────────────────────────────────────────────────────

    def epoch_start(self):
        """Call at the beginning of each training epoch."""
        if self._current_epoch == 0:
            self._run_id = str(uuid.uuid4())[:8]
            self._profiler.start()
        self._epoch_t0 = time.time()

    def epoch_end(self):
        """Call at the end of each training epoch."""
        self._current_epoch += 1
        epoch_hw = self._profiler.drain_epoch()
        epoch_energy = self._energy_calc.calculate(epoch_hw)
        epoch_carbon = self._carbon_est.estimate(epoch_energy.total_energy_kwh)

        ep = EpochResult(
            epoch=self._current_epoch,
            hardware=epoch_hw,
            energy=epoch_energy,
            carbon=epoch_carbon,
        )
        self._epoch_results.append(ep)

        if self._current_epoch <= self.monitor_epochs:
            self._monitored_energy_kwh += epoch_energy.total_energy_kwh
            self._monitored_duration_s += epoch_hw.duration_s

        if self.verbose >= 1:
            print(ep.summary())
            if self._current_epoch == self.monitor_epochs:
                self._print_prediction()

    def stop(self):
        """Finalize tracking (use after epoch loop or for early stopping)."""
        hw = self._profiler.stop()
        # If epochs were used, build aggregate from epoch results; otherwise use raw profile
        if self._epoch_results:
            hw = self._aggregate_epoch_hardware()
        self._finalize(hw)
        if self.verbose >= 1:
            print("\n" + self.result.summary())
        self._write_log()

    # ── context manager interface ─────────────────────────────────────────────

    def __enter__(self) -> "CarbonTracker":
        self._run_id = str(uuid.uuid4())[:8]
        self._profiler.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        hw = self._profiler.stop()
        self._finalize(hw)
        if self.verbose >= 1:
            print("\n" + self.result.summary())
        self._write_log()
        return False

    # ── decorator interface ───────────────────────────────────────────────────

    @classmethod
    def track(cls, **tracker_kwargs) -> Callable:
        """Decorator: @CarbonTracker.track(experiment='exp', model='GPT2')"""
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with cls(**tracker_kwargs) as ct:
                    result = fn(*args, **kwargs)
                return result
            return wrapper
        return decorator

    # ── prediction ────────────────────────────────────────────────────────────

    def _print_prediction(self):
        if self._monitored_duration_s <= 0:
            return
        remaining_epochs = self.epochs - self.monitor_epochs
        energy_per_epoch = self._monitored_energy_kwh / self.monitor_epochs
        pred_total_kwh   = self._monitored_energy_kwh + energy_per_epoch * remaining_epochs
        pred_carbon      = self._carbon_est.estimate(pred_total_kwh)
        pred_duration_s  = (self._monitored_duration_s / self.monitor_epochs) * self.epochs

        h = int(pred_duration_s // 3600)
        m = int((pred_duration_s % 3600) // 60)
        s = int(pred_duration_s % 60)

        print(f"\n  [Predicted for {self.epochs} epochs]")
        print(f"    Time:    {h:02d}:{m:02d}:{s:02d}")
        print(f"    Energy:  {pred_total_kwh*1000:.4f} Wh")
        print(f"    Carbon:  {pred_carbon.carbon_g:.6f} gCO2e")
        print(f"    Equiv:   {pred_carbon.equivalents['km_driven']:.6f} km driven\n")

    # ── internals ─────────────────────────────────────────────────────────────

    def _finalize(self, hw: HardwareProfile):
        energy = self._energy_calc.calculate(hw)
        carbon = self._carbon_est.estimate(energy.total_energy_kwh)
        record = self._cost_tracker.record_run(
            run_id=self._run_id,
            experiment_name=self.experiment_name,
            model_name=self.model_name,
            duration_s=hw.duration_s,
            energy_result=energy,
            carbon_result=carbon,
            tags=self.tags,
            metadata=self.metadata,
            user_name=self._user_name,
            organization=self._organization,
            user_region=self._user_region,
            user_region_source=self._user_region_source,
        )
        self.result = TrackingResult(
            run_id=self._run_id,
            experiment_name=self.experiment_name,
            model_name=self.model_name,
            hardware=hw,
            energy=energy,
            carbon=carbon,
            record=record,
            epoch_results=self._epoch_results,
        )

    def _aggregate_epoch_hardware(self) -> HardwareProfile:
        """Rebuild a single HardwareProfile from all epoch profiles."""
        eps = self._epoch_results
        n = len(eps)
        total_dur = sum(e.hardware.duration_s for e in eps)

        def wavg(attr):
            weights = [e.hardware.duration_s for e in eps]
            total_w = sum(weights)
            if total_w == 0:
                return 0.0
            return sum(getattr(e.hardware, attr) * w for e, w in zip(eps, weights)) / total_w

        last = eps[-1].hardware
        return HardwareProfile(
            duration_s=total_dur,
            avg_cpu_percent=wavg("avg_cpu_percent"),
            avg_cpu_power_w=wavg("avg_cpu_power_w"),
            peak_cpu_power_w=max(e.hardware.peak_cpu_power_w for e in eps),
            avg_dram_power_w=wavg("avg_dram_power_w"),
            avg_ram_gb=wavg("avg_ram_gb"),
            num_gpus=last.num_gpus,
            gpu_names=last.gpu_names,
            avg_gpu_utilization=wavg("avg_gpu_utilization"),
            avg_gpu_power_w=wavg("avg_gpu_power_w"),
            peak_gpu_power_w=max(e.hardware.peak_gpu_power_w for e in eps),
            total_cpu_energy_kwh=sum(e.hardware.total_cpu_energy_kwh for e in eps),
            total_gpu_energy_kwh=sum(e.hardware.total_gpu_energy_kwh for e in eps),
            total_dram_energy_kwh=sum(e.hardware.total_dram_energy_kwh for e in eps),
            cpu_model=last.cpu_model,
            cpu_power_source=last.cpu_power_source,
        )

    def _write_log(self):
        if not self.log_dir or not self.result:
            return
        import os, json
        from datetime import datetime, timezone
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"carbontracker_{self._run_id}_{ts}.log")
        r = self.result
        entry = {
            "run_id": r.run_id,
            "experiment": r.experiment_name,
            "model": r.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actual": {
                "duration_s": r.hardware.duration_s,
                "energy_kwh": r.energy.total_energy_kwh,
                "carbon_g": r.carbon.carbon_g,
                "carbon_kg": r.carbon.carbon_kg,
                "cost_usd": r.record.total_cost_usd,
            },
            "components": {
                "cpu": {
                    "model": r.hardware.cpu_model,
                    "power_source": r.hardware.cpu_power_source,
                    "avg_power_w": r.hardware.avg_cpu_power_w,
                    "energy_kwh": r.hardware.total_cpu_energy_kwh,
                },
                "gpu": {
                    "names": r.hardware.gpu_names,
                    "avg_power_w": r.hardware.avg_gpu_power_w,
                    "energy_kwh": r.hardware.total_gpu_energy_kwh,
                },
                "dram": {
                    "avg_power_w": r.hardware.avg_dram_power_w,
                    "energy_kwh": r.hardware.total_dram_energy_kwh,
                },
            },
            "location": {
                "region": r.carbon.region,
                "grid_intensity_gco2_kwh": r.carbon.grid_intensity_gco2_per_kwh,
                "intensity_source": r.carbon.intensity_source,
            },
            "equivalents": r.carbon.equivalents,
            "epochs": [
                {
                    "epoch": e.epoch,
                    "duration_s": e.hardware.duration_s,
                    "energy_kwh": e.energy.total_energy_kwh,
                    "carbon_g": e.carbon.carbon_g,
                }
                for e in r.epoch_results
            ],
            "tags": self.tags,
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker
