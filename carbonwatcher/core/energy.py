"""Energy consumption calculations from hardware profiles."""

from dataclasses import dataclass
from carbonwatcher.core.hardware import HardwareProfile


# PUE (Power Usage Effectiveness) by infrastructure type
PUE_VALUES = {
    "cloud":      1.10,   # modern hyperscaler (Google/AWS average)
    "on_premise": 1.50,   # typical enterprise data center
    "laptop":     1.00,   # no overhead — battery-powered
    "workstation": 1.05,  # local desktop
}

# Cloud GPU on-demand pricing (USD per GPU-hour, approximate 2025 rates)
CLOUD_GPU_PRICES = {
    "A100":    2.21,
    "V100":    0.90,
    "T4":      0.35,
    "RTX 4090": 0.74,
    "default": 0.50,
}

CLOUD_CPU_PRICE_PER_VCPU_HOUR = 0.048  # e.g., AWS c6i


@dataclass
class EnergyResult:
    raw_energy_kwh: float       # energy drawn by hardware
    pue_overhead: float         # PUE multiplier used
    total_energy_kwh: float     # raw × PUE
    infrastructure: str

    # cost breakdown
    compute_cost_usd: float
    storage_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.compute_cost_usd + self.storage_cost_usd


class EnergyCalculator:
    def __init__(
        self,
        infrastructure: str = "cloud",
        num_vcpus: int = 8,
        gpu_model: str = "default",
        storage_gb: float = 0.0,
        custom_pue: float | None = None,
        custom_gpu_price: float | None = None,
        custom_cpu_price: float | None = None,
    ):
        if infrastructure not in PUE_VALUES and custom_pue is None:
            raise ValueError(f"Unknown infrastructure '{infrastructure}'. "
                             f"Choose from: {list(PUE_VALUES.keys())} or pass custom_pue.")
        self.infrastructure = infrastructure
        self.num_vcpus = num_vcpus
        self.gpu_model = gpu_model
        self.storage_gb = storage_gb
        self.pue = custom_pue if custom_pue is not None else PUE_VALUES[infrastructure]
        self.gpu_price = custom_gpu_price if custom_gpu_price is not None else (
            CLOUD_GPU_PRICES.get(gpu_model, CLOUD_GPU_PRICES["default"])
        )
        self.cpu_price = custom_cpu_price if custom_cpu_price is not None else CLOUD_CPU_PRICE_PER_VCPU_HOUR

    def calculate(self, profile: HardwareProfile) -> EnergyResult:
        raw_kwh = profile.total_energy_kwh
        total_kwh = raw_kwh * self.pue

        hours = profile.duration_s / 3600.0

        # GPU cost: price per GPU-hour × number of GPUs × hours
        gpu_cost = self.gpu_price * max(profile.num_gpus, 1) * hours if profile.num_gpus > 0 else 0.0

        # CPU cost: price per vCPU-hour × vCPUs × hours × utilization ratio
        # Only charge for CPU time when no GPU (avoid double-charging cloud instance)
        cpu_cost = 0.0
        if profile.num_gpus == 0:
            cpu_cost = self.cpu_price * self.num_vcpus * hours

        # Storage cost: $0.023/GB/month → per-second rate
        storage_cost = self.storage_gb * 0.023 / (30 * 24 * 3600) * profile.duration_s

        return EnergyResult(
            raw_energy_kwh=raw_kwh,
            pue_overhead=self.pue,
            total_energy_kwh=total_kwh,
            infrastructure=self.infrastructure,
            compute_cost_usd=gpu_cost + cpu_cost,
            storage_cost_usd=storage_cost,
        )
