"""
User location detection and configuration.

Priority order:
  1. Config file  (~/.carbonwatcher/config.json)  — set once via CLI
  2. IP geolocation  (ipapi.co — free, no key needed)
  3. Default  (us-east-1 — always works, prints a notice)

The config file stores:
  - user_region:   where YOUR machine is (for local compute carbon)
  - user_name:     for team reports
  - organization:  for team grouping
"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional

from carbonwatcher.core.carbon import GRID_INTENSITY

_CONFIG_DIR  = os.path.expanduser("~/.carbonwatcher")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.json")
_GEO_API     = "https://ipapi.co/json/"
_GEO_TIMEOUT = 3.0

# Map country names / codes returned by ipapi → our GRID_INTENSITY keys
_COUNTRY_TO_REGION: dict[str, str] = {
    "IN": "India",
    "US": "USA",
    "DE": "Germany",
    "FR": "France",
    "GB": "UK",
    "CN": "China",
    "JP": "Japan",
    "CA": "Canada",
    "BR": "Brazil",
    "AU": "Australia",
    # Add more as needed
}


# ── Config file ───────────────────────────────────────────────────────────────

def load_config() -> dict:
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(updates: dict):
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    config = load_config()
    config.update(updates)
    with open(_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_config_value(key: str, default=None):
    return load_config().get(key, default)


# ── IP geolocation ────────────────────────────────────────────────────────────

def _fetch_geo() -> Optional[dict]:
    try:
        req = urllib.request.Request(_GEO_API, headers={"User-Agent": "carbonwatcher/0.2"})
        with urllib.request.urlopen(req, timeout=_GEO_TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def detect_location() -> tuple[str, str]:
    """
    Detect the user's country and map to a grid region key.
    Returns (region_key, source) where source is 'config'|'geoip'|'default'.
    """
    # 1. Config file
    config_region = get_config_value("user_region")
    if config_region:
        return config_region, "config"

    # 2. IP geolocation
    geo = _fetch_geo()
    if geo:
        country_code = geo.get("country_code", "")
        country_name = geo.get("country_name", "")
        city         = geo.get("city", "")
        region_name  = geo.get("region", "")

        # Try country code mapping first
        region = _COUNTRY_TO_REGION.get(country_code)
        if region and region in GRID_INTENSITY:
            return region, f"geoip:{country_name}"

        # Try direct country name match
        if country_name in GRID_INTENSITY:
            return country_name, f"geoip:{country_name}"

    # 3. Default
    return "us-east-1", "default"


def get_user_region(override: Optional[str] = None) -> tuple[str, str]:
    """
    Get the region to use for local compute carbon.
    override takes priority over everything else.
    """
    if override:
        if override not in GRID_INTENSITY:
            raise ValueError(
                f"Unknown region '{override}'. "
                f"Known: {list(GRID_INTENSITY.keys())}"
            )
        return override, "override"
    return detect_location()
