"""Reduction goals — set targets, track progress, get alerts."""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import Progress, BarColumn, TextColumn

from carbonwatcher.core.location import _CONFIG_DIR, load_config, save_config
from carbonwatcher.dashboard.monthly import combined_monthly

console = Console()

_GOALS_FILE = os.path.join(_CONFIG_DIR, "goals.json")


# ── Goals persistence ─────────────────────────────────────────────────────────

def load_goals() -> dict:
    if os.path.exists(_GOALS_FILE):
        try:
            with open(_GOALS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_goals(goals: dict):
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(_GOALS_FILE, "w") as f:
        json.dump(goals, f, indent=2)


def set_goal(
    period: str,          # "monthly" | "yearly"
    carbon_g: float,      # target max gCO2e
    cost_usd: Optional[float] = None,
    note: str = "",
):
    """Set a carbon (and optional cost) reduction goal."""
    goals = load_goals()
    goals[period] = {
        "carbon_g":  carbon_g,
        "cost_usd":  cost_usd,
        "note":      note,
        "set_at":    datetime.now(timezone.utc).isoformat(),
    }
    save_goals(goals)
    console.print(f"[green]Goal set:[/green] max {carbon_g:.2f} gCO2e / {period}")


# ── Progress display ──────────────────────────────────────────────────────────

def print_goals_progress(
    training_ledger: str = "carbonwatcher_ledger.json",
    llm_ledger: str = "carbonwatcher_llm_ledger.json",
    user_name: Optional[str] = None,
    organization: Optional[str] = None,
):
    goals = load_goals()
    if not goals:
        console.print(
            "[yellow]No goals set.[/yellow] "
            "Set one with: [bold]carbonwatcher goals set --monthly 100[/bold]"
        )
        return

    data = combined_monthly(training_ledger, llm_ledger, user_name, organization)
    now = datetime.now(timezone.utc)
    current_month = now.strftime("%Y-%m")
    current_year  = now.strftime("%Y")

    # Current month totals
    month_carbon = data.get(current_month, {}).get("total_carbon_g", 0)
    month_cost   = data.get(current_month, {}).get("total_cost_usd", 0)

    # Current year totals
    year_carbon = sum(
        d["total_carbon_g"] for m, d in data.items() if m.startswith(current_year)
    )
    year_cost = sum(
        d["total_cost_usd"] for m, d in data.items() if m.startswith(current_year)
    )

    table = Table(title="Reduction Goals Progress", box=box.ROUNDED,
                  header_style="bold cyan")
    table.add_column("Period")
    table.add_column("Metric")
    table.add_column("Goal")
    table.add_column("Used")
    table.add_column("Remaining")
    table.add_column("Status")

    def _row(period, label, goal_val, used_val, unit):
        remaining = goal_val - used_val
        pct = min(used_val / goal_val * 100, 100) if goal_val > 0 else 0
        if pct >= 100:
            status = "[red]EXCEEDED[/red]"
        elif pct >= 80:
            status = f"[yellow]⚠ {pct:.1f}%[/yellow]"
        else:
            status = f"[green]✓ {pct:.1f}%[/green]"
        table.add_row(
            period, label,
            f"{goal_val:.2f} {unit}",
            f"{used_val:.4f} {unit}",
            f"{max(remaining, 0):.4f} {unit}",
            status,
        )

    if "monthly" in goals:
        g = goals["monthly"]
        _row(f"{current_month} (monthly)", "Carbon", g["carbon_g"], month_carbon, "gCO2e")
        if g.get("cost_usd"):
            _row(f"{current_month} (monthly)", "Cost", g["cost_usd"], month_cost, "USD")

    if "yearly" in goals:
        g = goals["yearly"]
        _row(f"{current_year} (yearly)", "Carbon", g["carbon_g"], year_carbon, "gCO2e")
        if g.get("cost_usd"):
            _row(f"{current_year} (yearly)", "Cost", g["cost_usd"], year_cost, "USD")

    console.print(table)

    # Month-by-month progress vs monthly goal
    if "monthly" in goals and data:
        target = goals["monthly"]["carbon_g"]
        console.print(f"\n[bold]Monthly carbon vs goal ({target:.2f} gCO2e)[/bold]")
        for month in sorted(data.keys()):
            carbon = data[month]["total_carbon_g"]
            pct = min(carbon / target, 1.0) if target > 0 else 0
            bar_len = int(pct * 30)
            color = "green" if pct < 0.8 else ("yellow" if pct < 1.0 else "red")
            bar = f"[{color}]{'█' * bar_len}{'░' * (30 - bar_len)}[/{color}]"
            console.print(f"  {month}  {bar}  {carbon:.4f} / {target:.2f} g")
