"""
Energy, carbon, and cost estimates for API-hosted LLM models.

Energy figures are derived from:
  - Patterson et al. 2021 (Google, "Carbon Footprint of ML")
  - Luccioni et al. 2023 ("Power Hungry Processing")
  - Samsi et al. 2023 (LLM inference benchmarks)
  - Provider sustainability reports (Google, Microsoft, Anthropic)

These are estimates — providers don't publish exact per-token power draws.
We model: energy_per_token = (GPU_power_W × inference_time_per_token_s) / num_parallel_tokens
collapsed to a single wh_per_1k_tokens figure validated against published
CO2 numbers where available.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelSpec:
    provider: str
    model_id: str               # canonical ID used in API calls
    display_name: str
    # Energy per 1000 tokens (Wh) — separate input/output because output
    # requires an additional forward pass (autoregressive generation)
    wh_per_1k_input_tokens: float
    wh_per_1k_output_tokens: float
    # Grid carbon intensity for the provider's data centers (gCO2/kWh)
    # Sources: Google CFR 2023, Microsoft SER 2023, AWS sustainability report
    provider_grid_intensity_gco2_kwh: float
    # Pricing (USD per 1M tokens) — update when providers change rates
    cost_per_1m_input_tokens: float
    cost_per_1m_output_tokens: float
    # Context window (tokens)
    context_window: int = 0
    notes: str = ""


# ── Anthropic (Claude) ────────────────────────────────────────────────────────
# Anthropic runs on AWS + Google Cloud; uses ~0.0003 kgCO2/kWh blended estimate
# Energy estimates cross-checked against Luccioni 2023 for similarly-sized models

ANTHROPIC_GRID = 200.0   # gCO2/kWh — blended AWS/GCP with renewable purchases

ANTHROPIC_MODELS: list[ModelSpec] = [
    ModelSpec(
        provider="anthropic", model_id="claude-opus-4-7",
        display_name="Claude Opus 4.7",
        wh_per_1k_input_tokens=0.0087,
        wh_per_1k_output_tokens=0.0348,
        provider_grid_intensity_gco2_kwh=ANTHROPIC_GRID,
        cost_per_1m_input_tokens=15.00,
        cost_per_1m_output_tokens=75.00,
        context_window=200_000,
    ),
    ModelSpec(
        provider="anthropic", model_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        wh_per_1k_input_tokens=0.0029,
        wh_per_1k_output_tokens=0.0116,
        provider_grid_intensity_gco2_kwh=ANTHROPIC_GRID,
        cost_per_1m_input_tokens=3.00,
        cost_per_1m_output_tokens=15.00,
        context_window=200_000,
    ),
    ModelSpec(
        provider="anthropic", model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        wh_per_1k_input_tokens=0.00058,
        wh_per_1k_output_tokens=0.00232,
        provider_grid_intensity_gco2_kwh=ANTHROPIC_GRID,
        cost_per_1m_input_tokens=0.80,
        cost_per_1m_output_tokens=4.00,
        context_window=200_000,
    ),
    # Legacy
    ModelSpec(
        provider="anthropic", model_id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        wh_per_1k_input_tokens=0.0029,
        wh_per_1k_output_tokens=0.0116,
        provider_grid_intensity_gco2_kwh=ANTHROPIC_GRID,
        cost_per_1m_input_tokens=3.00,
        cost_per_1m_output_tokens=15.00,
        context_window=200_000,
    ),
    ModelSpec(
        provider="anthropic", model_id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        wh_per_1k_input_tokens=0.00025,
        wh_per_1k_output_tokens=0.00125,
        provider_grid_intensity_gco2_kwh=ANTHROPIC_GRID,
        cost_per_1m_input_tokens=0.25,
        cost_per_1m_output_tokens=1.25,
        context_window=200_000,
    ),
]

# ── OpenAI ────────────────────────────────────────────────────────────────────
# OpenAI runs on Azure; Microsoft targets 100% renewable by 2025
# Energy estimates from Luccioni 2023 + OpenAI model card notes

OPENAI_GRID = 180.0  # gCO2/kWh — Azure with renewable commitments

OPENAI_MODELS: list[ModelSpec] = [
    ModelSpec(
        provider="openai", model_id="gpt-4o",
        display_name="GPT-4o",
        wh_per_1k_input_tokens=0.0031,
        wh_per_1k_output_tokens=0.0124,
        provider_grid_intensity_gco2_kwh=OPENAI_GRID,
        cost_per_1m_input_tokens=2.50,
        cost_per_1m_output_tokens=10.00,
        context_window=128_000,
    ),
    ModelSpec(
        provider="openai", model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        wh_per_1k_input_tokens=0.00062,
        wh_per_1k_output_tokens=0.00248,
        provider_grid_intensity_gco2_kwh=OPENAI_GRID,
        cost_per_1m_input_tokens=0.15,
        cost_per_1m_output_tokens=0.60,
        context_window=128_000,
    ),
    ModelSpec(
        provider="openai", model_id="gpt-4-turbo",
        display_name="GPT-4 Turbo",
        wh_per_1k_input_tokens=0.0087,
        wh_per_1k_output_tokens=0.0348,
        provider_grid_intensity_gco2_kwh=OPENAI_GRID,
        cost_per_1m_input_tokens=10.00,
        cost_per_1m_output_tokens=30.00,
        context_window=128_000,
    ),
    ModelSpec(
        provider="openai", model_id="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        wh_per_1k_input_tokens=0.00058,
        wh_per_1k_output_tokens=0.00232,
        provider_grid_intensity_gco2_kwh=OPENAI_GRID,
        cost_per_1m_input_tokens=0.50,
        cost_per_1m_output_tokens=1.50,
        context_window=16_385,
    ),
    ModelSpec(
        provider="openai", model_id="o1",
        display_name="o1",
        wh_per_1k_input_tokens=0.0087,
        wh_per_1k_output_tokens=0.0696,   # reasoning models generate many more tokens
        provider_grid_intensity_gco2_kwh=OPENAI_GRID,
        cost_per_1m_input_tokens=15.00,
        cost_per_1m_output_tokens=60.00,
        context_window=200_000,
        notes="Reasoning model; output tokens include chain-of-thought",
    ),
    ModelSpec(
        provider="openai", model_id="o3-mini",
        display_name="o3-mini",
        wh_per_1k_input_tokens=0.0044,
        wh_per_1k_output_tokens=0.0348,
        provider_grid_intensity_gco2_kwh=OPENAI_GRID,
        cost_per_1m_input_tokens=1.10,
        cost_per_1m_output_tokens=4.40,
        context_window=200_000,
        notes="Reasoning model",
    ),
]

# ── Google (Gemini) ───────────────────────────────────────────────────────────
# Google targets carbon-free energy by 2030; current grid ~30% clean
GOOGLE_GRID = 300.0  # gCO2/kWh — GCP blended

GOOGLE_MODELS: list[ModelSpec] = [
    ModelSpec(
        provider="google", model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        wh_per_1k_input_tokens=0.00031,
        wh_per_1k_output_tokens=0.00124,
        provider_grid_intensity_gco2_kwh=GOOGLE_GRID,
        cost_per_1m_input_tokens=0.10,
        cost_per_1m_output_tokens=0.40,
        context_window=1_048_576,
    ),
    ModelSpec(
        provider="google", model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        wh_per_1k_input_tokens=0.0044,
        wh_per_1k_output_tokens=0.0174,
        provider_grid_intensity_gco2_kwh=GOOGLE_GRID,
        cost_per_1m_input_tokens=3.50,
        cost_per_1m_output_tokens=10.50,
        context_window=2_097_152,
    ),
    ModelSpec(
        provider="google", model_id="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        wh_per_1k_input_tokens=0.00044,
        wh_per_1k_output_tokens=0.00174,
        provider_grid_intensity_gco2_kwh=GOOGLE_GRID,
        cost_per_1m_input_tokens=0.075,
        cost_per_1m_output_tokens=0.30,
        context_window=1_048_576,
    ),
]

# ── Mistral ───────────────────────────────────────────────────────────────────
MISTRAL_GRID = 85.0  # gCO2/kWh — France-hosted, nuclear-heavy grid

MISTRAL_MODELS: list[ModelSpec] = [
    ModelSpec(
        provider="mistral", model_id="mistral-large-latest",
        display_name="Mistral Large",
        wh_per_1k_input_tokens=0.0029,
        wh_per_1k_output_tokens=0.0116,
        provider_grid_intensity_gco2_kwh=MISTRAL_GRID,
        cost_per_1m_input_tokens=3.00,
        cost_per_1m_output_tokens=9.00,
        context_window=131_072,
        notes="Hosted in France — very low-carbon grid",
    ),
    ModelSpec(
        provider="mistral", model_id="mistral-small-latest",
        display_name="Mistral Small",
        wh_per_1k_input_tokens=0.00058,
        wh_per_1k_output_tokens=0.00232,
        provider_grid_intensity_gco2_kwh=MISTRAL_GRID,
        cost_per_1m_input_tokens=0.20,
        cost_per_1m_output_tokens=0.60,
        context_window=131_072,
    ),
]

# ── Master registry ───────────────────────────────────────────────────────────
_ALL_MODELS: dict[str, ModelSpec] = {
    m.model_id: m
    for m in ANTHROPIC_MODELS + OPENAI_MODELS + GOOGLE_MODELS + MISTRAL_MODELS
}

# Prefix aliases so partial names resolve (e.g. "claude-opus" → opus spec)
_PREFIX_ALIASES: list[tuple[str, str]] = [
    ("claude-opus-4",       "claude-opus-4-7"),
    ("claude-sonnet-4",     "claude-sonnet-4-6"),
    ("claude-haiku-4",      "claude-haiku-4-5-20251001"),
    ("claude-3-5-sonnet",   "claude-3-5-sonnet-20241022"),
    ("claude-3-haiku",      "claude-3-haiku-20240307"),
    ("gemini-2.0-flash",    "gemini-2.0-flash"),
    ("gemini-flash",        "gemini-1.5-flash"),
    ("gemini-pro",          "gemini-1.5-pro"),
]


def lookup_model(model_id: str) -> Optional[ModelSpec]:
    """Exact match first, then prefix aliases, then None."""
    if model_id in _ALL_MODELS:
        return _ALL_MODELS[model_id]
    for prefix, canonical in _PREFIX_ALIASES:
        if model_id.startswith(prefix):
            return _ALL_MODELS.get(canonical)
    return None


def list_models(provider: Optional[str] = None) -> list[ModelSpec]:
    models = list(_ALL_MODELS.values())
    if provider:
        models = [m for m in models if m.provider == provider.lower()]
    return sorted(models, key=lambda m: (m.provider, m.model_id))
