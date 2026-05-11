"""Carbon footprint estimation using grid carbon intensity data."""

from dataclasses import dataclass, field

# Carbon intensity in gCO2eq/kWh by region (2024 averages from Electricity Maps / IEA)
GRID_INTENSITY = {
    # Cloud regions → real-world grid
    "us-east-1":      386.0,   # US East (Virginia)  — PJM grid
    "us-east-2":      480.0,   # US East (Ohio)       — MISO
    "us-west-1":      270.0,   # US West (N. California)
    "us-west-2":      130.0,   # US West (Oregon)     — heavy hydro
    "eu-west-1":       88.0,   # Ireland              — wind-heavy
    "eu-central-1":  311.0,    # Frankfurt            — coal still in mix
    "ap-southeast-1": 408.0,   # Singapore
    "ap-northeast-1": 465.0,   # Tokyo
    "ap-south-1":     708.0,   # Mumbai               — coal-heavy
    "ca-central-1":   130.0,   # Canada               — hydro-heavy
    "sa-east-1":       74.0,   # São Paulo            — hydro
    # Countries
    "USA":            386.0,
    "Germany":        311.0,
    "France":          58.0,   # nuclear-heavy
    "UK":             148.0,
    "China":          555.0,
    "India":          708.0,
    "Canada":         130.0,
    "Brazil":          74.0,
    "Australia":      560.0,
    "Japan":          465.0,
    # Renewable / special
    "renewable":        0.0,
    "100%_renewable":   0.0,
    "nuclear":          5.5,
    "coal":           820.0,
}

# Social cost of carbon: ~$185/tonne CO2e (US EPA 2023 revised)
SOCIAL_COST_PER_TONNE = 185.0


@dataclass
class CarbonResult:
    energy_kwh: float
    region: str
    grid_intensity_gco2_per_kwh: float
    carbon_kg: float
    carbon_g: float
    equivalents: dict                               # human-readable comparisons
    intensity_source: str = field(default="static_table")  # where the intensity came from

    @property
    def social_cost_usd(self) -> float:
        return (self.carbon_kg / 1000.0) * SOCIAL_COST_PER_TONNE

    @property
    def is_low_carbon(self) -> bool:
        return self.grid_intensity_gco2_per_kwh < 200.0


class CarbonEstimator:
    def __init__(self, region: str = "us-east-1", custom_intensity: float | None = None):
        # Defer import to avoid circular dependency
        from carbonwatcher.core.grid import GridIntensityResolver
        self.region = region
        self._resolver = GridIntensityResolver(region=region, custom_intensity=custom_intensity)
        # Resolve once at construction so errors surface early
        self.intensity, self._source = self._resolver.resolve()

    def estimate(self, energy_kwh: float) -> CarbonResult:
        # Re-resolve each call so live API data stays fresh across long runs
        intensity, source = self._resolver.resolve()
        self.intensity = intensity

        carbon_g = energy_kwh * intensity
        carbon_kg = carbon_g / 1000.0

        return CarbonResult(
            energy_kwh=energy_kwh,
            region=self.region,
            grid_intensity_gco2_per_kwh=intensity,
            carbon_kg=carbon_kg,
            carbon_g=carbon_g,
            equivalents=_compute_equivalents(carbon_g),
            intensity_source=source,
        )

    @staticmethod
    def list_regions() -> dict:
        return dict(sorted(GRID_INTENSITY.items(), key=lambda x: x[1]))


def _compute_equivalents(carbon_g: float) -> dict:
    """Express carbon emissions in relatable everyday terms."""
    km_driven = carbon_g / 192.0            # avg car: 192 gCO2/km
    smartphone_charges = carbon_g / 8.22    # 8.22 g per charge cycle
    led_hours = carbon_g / 2.17             # 10W LED: ~2.17 gCO2/hr (US grid)
    trees_days = carbon_g / 21_900.0 * 365  # tree absorbs ~21.9 kgCO2/yr

    return {
        "km_driven": round(km_driven, 4),
        "smartphone_charges": round(smartphone_charges, 2),
        "led_bulb_hours": round(led_hours, 2),
        "tree_absorption_days": round(trees_days, 4),
    }
