"""Hardware profiler — samples CPU, GPU, and DRAM power in a background thread.

Power measurement priority:
  CPU:  1) Intel RAPL (direct hardware)  2) Apple powermetrics  3) TDP estimate
  GPU:  NVIDIA NVML (direct hardware)
  DRAM: psutil RSS scaled by DRAM TDP/capacity ratio
"""

import os
import re
import glob
import time
import threading
import platform
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

import psutil

# ── NVIDIA NVML (optional) ────────────────────────────────────────────────────
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
    _GPU_COUNT = pynvml.nvmlDeviceGetCount()
except Exception:
    _NVML_AVAILABLE = False
    _GPU_COUNT = 0

# ── Intel RAPL paths ──────────────────────────────────────────────────────────
_RAPL_BASE = "/sys/class/powercap/intel-rapl"
_RAPL_PACKAGE_GLOB = "/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj"
_RAPL_DRAM_GLOB    = "/sys/class/powercap/intel-rapl/intel-rapl:*/intel-rapl:*:*/energy_uj"

# ── TDP fallbacks (watts) ─────────────────────────────────────────────────────
_CPU_TDP_FALLBACK = 65.0
_DRAM_POWER_W_PER_GB = 0.375   # ~3W per 8 GB DIMM

_GPU_TDP_FALLBACKS = {
    "A100": 400.0, "H100": 700.0, "V100": 300.0, "T4": 70.0,
    "RTX 4090": 450.0, "RTX 3090": 350.0, "RTX 3080": 320.0,
    "RTX 3070": 220.0, "RTX 2080": 215.0, "M1": 10.0,
}


# ── RAPL helpers ──────────────────────────────────────────────────────────────

def _rapl_available() -> bool:
    return os.path.isdir(_RAPL_BASE)


def _read_rapl_uj(glob_pattern: str) -> float:
    """Sum energy_uj across all matched RAPL domains."""
    total = 0.0
    for path in glob.glob(glob_pattern):
        try:
            with open(path) as f:
                total += float(f.read())
        except OSError:
            pass
    return total


class _RaplReader:
    """Reads Intel RAPL package energy counters and converts delta → watts."""

    def __init__(self):
        self._last_uj: Optional[float] = None
        self._last_ts: Optional[float] = None

    def watts(self) -> Optional[float]:
        """Return average CPU package power since last call, or None if unavailable."""
        if not _rapl_available():
            return None
        uj = _read_rapl_uj(_RAPL_PACKAGE_GLOB)
        ts = time.monotonic()
        if self._last_uj is None:
            self._last_uj, self._last_ts = uj, ts
            return None
        delta_uj = uj - self._last_uj
        delta_s  = ts - self._last_ts
        self._last_uj, self._last_ts = uj, ts
        if delta_s <= 0 or delta_uj < 0:
            return None
        return delta_uj / 1e6 / delta_s  # µJ → J → W

    def dram_watts(self) -> Optional[float]:
        """Return DRAM power from RAPL DRAM domain if available."""
        if not _rapl_available():
            return None
        paths = glob.glob(_RAPL_DRAM_GLOB)
        dram_paths = [p for p in paths if "dram" in p.lower()]
        if not dram_paths:
            return None
        # For DRAM we need two readings; use a simple cached approach
        uj = sum(float(open(p).read()) for p in dram_paths if os.path.exists(p))
        return uj  # raw — caller handles delta


# ── Apple Silicon powermetrics ────────────────────────────────────────────────

def _apple_silicon_watts() -> Optional[float]:
    """
    Sample CPU+GPU combined package power via powermetrics (requires sudo or
    the process already holding the privilege). Returns None if unavailable.
    """
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.check_output(
            ["powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"],
            timeout=2, stderr=subprocess.DEVNULL,
        ).decode(errors="replace")
        # Look for "CPU Power: 123 mW" or "Package Power: 123 mW"
        for pat in (r"CPU Power:\s+([\d.]+)\s*mW", r"Package Power:\s+([\d.]+)\s*mW"):
            m = re.search(pat, out)
            if m:
                return float(m.group(1)) / 1000.0   # mW → W
    except Exception:
        pass
    return None


# ── CPU model detection ───────────────────────────────────────────────────────

def _detect_cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                timeout=1, stderr=subprocess.DEVNULL,
            ).decode().strip()
            if out:
                return out
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.model"],
                timeout=1, stderr=subprocess.DEVNULL,
            ).decode().strip()
            return out or platform.processor() or "Apple Silicon"
        except Exception:
            pass
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def _cpu_tdp_estimate() -> float:
    """Estimate current CPU power from utilization × TDP fallback."""
    util = psutil.cpu_percent(interval=None) / 100.0
    idle_fraction = 0.30
    return _CPU_TDP_FALLBACK * (idle_fraction + (1 - idle_fraction) * util)


# ── GPU sampling ──────────────────────────────────────────────────────────────

def _get_gpu_samples() -> List[dict]:
    if not _NVML_AVAILABLE:
        return []
    # Respect CUDA_VISIBLE_DEVICES if set
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != "":
        try:
            indices = [int(x) for x in visible.split(",") if x.strip().lstrip("-").isdigit()]
        except ValueError:
            indices = list(range(_GPU_COUNT))
    else:
        indices = list(range(_GPU_COUNT))

    samples = []
    for i in indices:
        if i < 0 or i >= _GPU_COUNT:
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except pynvml.NVMLError:
            tdp = next((v for k, v in _GPU_TDP_FALLBACKS.items() if k in name), 200.0)
            power_w = tdp * (0.30 + 0.70 * util / 100.0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        samples.append({
            "index": i,
            "name": name,
            "utilization": util,
            "power_w": power_w,
            "mem_used_gb": mem_info.used / 1e9,
            "mem_total_gb": mem_info.total / 1e9,
        })
    return samples


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class HardwareSample:
    timestamp: float
    cpu_power_w: float
    cpu_percent: float
    dram_power_w: float
    ram_used_gb: float
    gpu_samples: List[dict] = field(default_factory=list)
    cpu_source: str = "tdp_estimate"   # "rapl" | "powermetrics" | "tdp_estimate"


@dataclass
class HardwareProfile:
    """Aggregated hardware stats for a completed run (or epoch)."""
    duration_s: float
    avg_cpu_percent: float
    avg_cpu_power_w: float
    peak_cpu_power_w: float
    avg_dram_power_w: float
    avg_ram_gb: float
    num_gpus: int
    gpu_names: List[str]
    avg_gpu_utilization: float
    avg_gpu_power_w: float
    peak_gpu_power_w: float
    total_cpu_energy_kwh: float
    total_gpu_energy_kwh: float
    total_dram_energy_kwh: float
    cpu_model: str = ""
    cpu_power_source: str = "tdp_estimate"

    @property
    def total_energy_kwh(self) -> float:
        return self.total_cpu_energy_kwh + self.total_gpu_energy_kwh + self.total_dram_energy_kwh

    @property
    def gpu_label(self) -> str:
        if not self.gpu_names:
            return "None detected"
        unique = list(dict.fromkeys(self.gpu_names))
        if len(unique) == 1 and len(self.gpu_names) > 1:
            return f"{len(self.gpu_names)}× {unique[0]}"
        return ", ".join(unique)


# ── Profiler ──────────────────────────────────────────────────────────────────

class HardwareProfiler:
    """Continuously samples hardware metrics in a background thread."""

    def __init__(self, sample_interval_s: float = 1.0):
        self.sample_interval_s = sample_interval_s
        self._samples: List[HardwareSample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cpu_model: str = ""
        self._rapl = _RaplReader()
        self._apple_watts_cached: Optional[float] = None

    def start(self):
        self._stop_event.clear()
        self._samples.clear()
        self._cpu_model = _detect_cpu_model()
        psutil.cpu_percent(interval=None)   # warm up
        # Prime RAPL reader
        self._rapl.watts()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> HardwareProfile:
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        return self._build_profile(self._samples)

    def snapshot(self) -> HardwareProfile:
        """Return a profile of samples collected so far (non-destructive)."""
        return self._build_profile(list(self._samples))

    def drain_epoch(self) -> HardwareProfile:
        """Return profile of samples collected since last drain, then reset."""
        samples = list(self._samples)
        self._samples.clear()
        return self._build_profile(samples)

    def _sample_loop(self):
        while not self._stop_event.is_set():
            sample = self._take_sample()
            self._samples.append(sample)
            time.sleep(self.sample_interval_s)

    def _take_sample(self) -> HardwareSample:
        ts = time.time()
        cpu_pct = psutil.cpu_percent(interval=None)

        # CPU power — priority: RAPL > Apple powermetrics > TDP estimate
        rapl_w = self._rapl.watts()
        if rapl_w is not None:
            cpu_power_w = rapl_w
            cpu_source = "rapl"
        else:
            apple_w = _apple_silicon_watts()
            if apple_w is not None:
                cpu_power_w = apple_w
                cpu_source = "powermetrics"
            else:
                cpu_power_w = _cpu_tdp_estimate()
                cpu_source = "tdp_estimate"

        # DRAM power — from psutil memory usage × empirical W/GB
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / 1e9
        dram_power_w = ram_used_gb * _DRAM_POWER_W_PER_GB

        return HardwareSample(
            timestamp=ts,
            cpu_power_w=cpu_power_w,
            cpu_percent=cpu_pct,
            dram_power_w=dram_power_w,
            ram_used_gb=ram_used_gb,
            gpu_samples=_get_gpu_samples(),
            cpu_source=cpu_source,
        )

    def _build_profile(self, samples: List[HardwareSample]) -> HardwareProfile:
        gpu_names: List[str] = []
        if samples:
            seen = {}
            for s in samples:
                for g in s.gpu_samples:
                    seen[g["index"]] = g["name"]
            gpu_names = [seen[i] for i in sorted(seen)]

        empty = HardwareProfile(
            duration_s=0, avg_cpu_percent=0, avg_cpu_power_w=0, peak_cpu_power_w=0,
            avg_dram_power_w=0, avg_ram_gb=0, num_gpus=_GPU_COUNT, gpu_names=gpu_names,
            avg_gpu_utilization=0, avg_gpu_power_w=0, peak_gpu_power_w=0,
            total_cpu_energy_kwh=0, total_gpu_energy_kwh=0, total_dram_energy_kwh=0,
            cpu_model=self._cpu_model,
        )
        if not samples:
            return empty

        n = len(samples)
        duration = samples[-1].timestamp - samples[0].timestamp
        duration = max(duration, self.sample_interval_s)

        avg_cpu_pct  = sum(s.cpu_percent for s in samples) / n
        avg_cpu_pow  = sum(s.cpu_power_w for s in samples) / n
        peak_cpu_pow = max(s.cpu_power_w for s in samples)
        avg_dram_pow = sum(s.dram_power_w for s in samples) / n
        avg_ram      = sum(s.ram_used_gb for s in samples) / n
        cpu_source   = samples[-1].cpu_source

        cpu_energy_kwh  = avg_cpu_pow  * duration / 3_600_000
        dram_energy_kwh = avg_dram_pow * duration / 3_600_000

        gpu_utils, gpu_powers = [], []
        for s in samples:
            for g in s.gpu_samples:
                gpu_utils.append(g["utilization"])
                gpu_powers.append(g["power_w"])

        avg_gpu_util  = sum(gpu_utils)  / len(gpu_utils)  if gpu_utils  else 0.0
        avg_gpu_pow   = sum(gpu_powers) / len(gpu_powers) if gpu_powers else 0.0
        peak_gpu_pow  = max(gpu_powers) if gpu_powers else 0.0
        gpu_energy_kwh = avg_gpu_pow * duration / 3_600_000

        return HardwareProfile(
            duration_s=duration,
            avg_cpu_percent=avg_cpu_pct,
            avg_cpu_power_w=avg_cpu_pow,
            peak_cpu_power_w=peak_cpu_pow,
            avg_dram_power_w=avg_dram_pow,
            avg_ram_gb=avg_ram,
            num_gpus=_GPU_COUNT,
            gpu_names=gpu_names,
            avg_gpu_utilization=avg_gpu_util,
            avg_gpu_power_w=avg_gpu_pow,
            peak_gpu_power_w=peak_gpu_pow,
            total_cpu_energy_kwh=cpu_energy_kwh,
            total_gpu_energy_kwh=gpu_energy_kwh,
            total_dram_energy_kwh=dram_energy_kwh,
            cpu_model=self._cpu_model,
            cpu_power_source=cpu_source,
        )
