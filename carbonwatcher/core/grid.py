"""
Live grid carbon intensity — Electricity Maps API with static-table fallback.

Priority order:
  1. Electricity Maps live API (requires ELECTRICITY_MAPS_API_KEY env var)
  2. Auto-detect cloud region from instance metadata (AWS / GCP / Azure)
  3. Static lookup table (always available, no network needed)
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import Optional

from carbonwatcher.core.carbon import GRID_INTENSITY

# Electricity Maps → AWS region mapping
# https://app.electricitymaps.com/map
_AWS_TO_EMAPS_ZONE = {
    "us-east-1":      "US-MIDA-PJM",
    "us-east-2":      "US-MIDW-MISO",
    "us-west-1":      "US-CAL-CISO",
    "us-west-2":      "US-NW-PACW",
    "eu-west-1":      "IE",
    "eu-west-2":      "GB",
    "eu-central-1":   "DE",
    "ap-southeast-1": "SG",
    "ap-northeast-1": "JP-TK",
    "ap-south-1":     "IN-SO",
    "ca-central-1":   "CA-ON",
    "sa-east-1":      "BR-CS",
}

_GCP_TO_EMAPS_ZONE = {
    "us-central1":    "US-MIDW-MISO",
    "us-east1":       "US-SE-SOCO",
    "us-east4":       "US-MIDA-PJM",
    "us-west1":       "US-NW-BPAT",
    "us-west2":       "US-CAL-LADWP",
    "europe-west1":   "BE",
    "europe-west4":   "NL",
    "asia-east1":     "TW",
    "asia-southeast1":"SG",
}

_EMAPS_API_URL = "https://api.electricitymap.org/v3/carbon-intensity/latest"
_METADATA_TIMEOUT_S = 0.5   # don't block training for >500ms
_API_TIMEOUT_S = 3.0

# Simple in-process cache: (zone, intensity, fetched_at)
_cache: dict[str, tuple[float, float]] = {}
_CACHE_TTL_S = 3600.0  # 1 hour


def _fetch_url(url: str, headers: dict, timeout: float) -> Optional[dict]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _detect_aws_region() -> Optional[str]:
    """IMDSv2: two-step token + metadata fetch."""
    token_url = "http://169.254.169.254/latest/api/token"
    token_req = urllib.request.Request(
        token_url,
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
    )
    try:
        with urllib.request.urlopen(token_req, timeout=_METADATA_TIMEOUT_S) as r:
            token = r.read().decode()
        region_url = "http://169.254.169.254/latest/meta-data/placement/region"
        region_req = urllib.request.Request(
            region_url, headers={"X-aws-ec2-metadata-token": token}
        )
        with urllib.request.urlopen(region_req, timeout=_METADATA_TIMEOUT_S) as r:
            return r.read().decode()
    except Exception:
        return None


def _detect_gcp_region() -> Optional[str]:
    data = _fetch_url(
        "http://metadata.google.internal/computeMetadata/v1/instance/zone",
        headers={"Metadata-Flavor": "Google"},
        timeout=_METADATA_TIMEOUT_S,
    )
    if isinstance(data, str):
        # format: "projects/123/zones/us-central1-a" → strip zone suffix
        zone = data.split("/")[-1]
        return "-".join(zone.split("-")[:-1])  # "us-central1"
    return None


def _detect_azure_region() -> Optional[str]:
    data = _fetch_url(
        "http://169.254.169.254/metadata/instance/compute/location?api-version=2021-02-01&format=text",
        headers={"Metadata": "true"},
        timeout=_METADATA_TIMEOUT_S,
    )
    return data if isinstance(data, str) else None


def auto_detect_region() -> Optional[str]:
    """
    Try to detect the current cloud region from instance metadata.
    Returns an AWS/GCP region string, or None if not on a cloud instance.
    Fast — each probe times out in 500ms.
    """
    region = _detect_aws_region()
    if region:
        return region
    region = _detect_gcp_region()
    if region:
        return region
    # Azure regions don't map cleanly to our table yet; skip
    return None


def fetch_live_intensity(zone: str, api_key: str) -> Optional[float]:
    """
    Call Electricity Maps API for real-time gCO2/kWh for `zone`.
    Returns None on any error (network, quota, bad zone).
    """
    now = time.time()
    if zone in _cache:
        intensity, fetched_at = _cache[zone]
        if now - fetched_at < _CACHE_TTL_S:
            return intensity

    data = _fetch_url(
        f"{_EMAPS_API_URL}?zone={zone}",
        headers={"auth-token": api_key},
        timeout=_API_TIMEOUT_S,
    )
    if data and "carbonIntensity" in data:
        intensity = float(data["carbonIntensity"])
        _cache[zone] = (intensity, now)
        return intensity
    return None


class GridIntensityResolver:
    """
    Resolves grid carbon intensity for a region using three sources
    in priority order: live API → auto-detect cloud region → static table.
    """

    def __init__(self, region: str = "us-east-1", custom_intensity: float | None = None):
        self.region = region
        self.custom_intensity = custom_intensity
        self._api_key = os.environ.get("ELECTRICITY_MAPS_API_KEY")
        self._source: str = "unknown"
        self._emaps_zone: Optional[str] = None

    def resolve(self) -> tuple[float, str]:
        """
        Returns (intensity_gco2_per_kwh, source_description).
        source is one of: 'live_api', 'auto_detected', 'static_table', 'custom'.
        """
        if self.custom_intensity is not None:
            return self.custom_intensity, "custom"

        # Attempt live Electricity Maps API
        if self._api_key:
            zone = (
                _AWS_TO_EMAPS_ZONE.get(self.region)
                or _GCP_TO_EMAPS_ZONE.get(self.region)
                or self.region  # user may have passed a zone directly
            )
            intensity = fetch_live_intensity(zone, self._api_key)
            if intensity is not None:
                self._emaps_zone = zone
                return intensity, f"live_api:{zone}"

        # Auto-detect cloud region and use its static value
        detected = auto_detect_region()
        if detected and detected in GRID_INTENSITY:
            return GRID_INTENSITY[detected], f"auto_detected:{detected}"

        # Fall back to static table
        if self.region in GRID_INTENSITY:
            return GRID_INTENSITY[self.region], "static_table"

        raise ValueError(
            f"Unknown region '{self.region}' and no live API key set. "
            f"Known regions: {list(GRID_INTENSITY.keys())} or set ELECTRICITY_MAPS_API_KEY."
        )
