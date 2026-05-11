"""Monthly emissions dashboard — aggregation, ASCII bar charts, JSON/CSV export."""

import json
import csv
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from carbonwatcher.core.cost import CostTracker, RunRecord
from carbonwatcher.llm.tracker import LLMTracker, LLMCallRecord

console = Console()

_BAR_WIDTH = 30   # max width of ASCII bar in chart


# ── Data aggregation ──────────────────────────────────────────────────────────

def _month_key(timestamp: str) -> str:
    """Convert ISO timestamp → 'YYYY-MM'."""
    try:
        return timestamp[:7]
    except Exception:
        return "unknown"


def monthly_training_stats(records: List[RunRecord]) -> dict[str, dict]:
    """Group training RunRecords by month, return stats per month."""
    by_month: dict[str, list] = defaultdict(list)
    for r in records:
        by_month[_month_key(r.timestamp)].append(r)

    result = {}
    for month, recs in sorted(by_month.items()):
        result[month] = {
            "runs":          len(recs),
            "energy_kwh":    sum(r.energy_kwh for r in recs),
            "carbon_kg":     sum(r.carbon_kg for r in recs),
            "carbon_g":      sum(r.carbon_kg * 1000 for r in recs),
            "cost_usd":      sum(r.total_cost_usd for r in recs),
            "duration_h":    sum(r.duration_s for r in recs) / 3600,
            "experiments":   list({r.experiment_name for r in recs}),
            "models":        list({r.model_name for r in recs}),
            "regions":       list({r.region for r in recs}),
        }
    return result


def monthly_llm_stats(records: List[LLMCallRecord]) -> dict[str, dict]:
    """Group LLM call records by month, return stats per month."""
    by_month: dict[str, list] = defaultdict(list)
    for r in records:
        by_month[_month_key(r.timestamp)].append(r)

    result = {}
    for month, recs in sorted(by_month.items()):
        result[month] = {
            "calls":          len(recs),
            "total_tokens":   sum(r.total_tokens for r in recs),
            "input_tokens":   sum(r.input_tokens for r in recs),
            "output_tokens":  sum(r.output_tokens for r in recs),
            "energy_wh":      sum(r.total_energy_wh for r in recs),
            "carbon_g":       sum(r.carbon_g for r in recs),
            "carbon_kg":      sum(r.carbon_g / 1000 for r in recs),
            "cost_usd":       sum(r.total_cost_usd for r in recs),
            "models":         list({r.model_id for r in recs}),
            "providers":      list({r.provider for r in recs}),
        }
    return result


def combined_monthly(
    training_ledger: str = "carbonwatcher_ledger.json",
    llm_ledger: str = "carbonwatcher_llm_ledger.json",
    user_name: Optional[str] = None,
    organization: Optional[str] = None,
) -> dict[str, dict]:
    """
    Merge training + LLM stats by month into one combined dict.
    Each month entry has: total_carbon_g, total_cost_usd, breakdown dict.
    """
    train_records = []
    if os.path.exists(training_ledger):
        ct = CostTracker(ledger_path=training_ledger)
        records = ct.get_all_runs()
        if user_name:
            records = [r for r in records if r.user_name == user_name]
        if organization:
            records = [r for r in records if r.organization == organization]
        train_records = records

    llm_records = []
    if os.path.exists(llm_ledger):
        lt = LLMTracker(ledger_path=llm_ledger)
        llm_records = lt.get_all_calls()

    train_by_month = monthly_training_stats(train_records)
    llm_by_month   = monthly_llm_stats(llm_records)

    all_months = sorted(set(list(train_by_month.keys()) + list(llm_by_month.keys())))
    combined = {}
    for month in all_months:
        t = train_by_month.get(month, {})
        l = llm_by_month.get(month, {})
        combined[month] = {
            "month":              month,
            "total_carbon_g":     t.get("carbon_g", 0) + l.get("carbon_g", 0),
            "total_cost_usd":     t.get("cost_usd", 0) + l.get("cost_usd", 0),
            "total_energy_kwh":   t.get("energy_kwh", 0) + l.get("energy_wh", 0) / 1000,
            "training":           t,
            "llm":                l,
        }
    return combined


# ── ASCII bar chart ───────────────────────────────────────────────────────────

def _bar(value: float, max_value: float, width: int = _BAR_WIDTH) -> str:
    if max_value <= 0:
        return " " * width
    filled = int(round(value / max_value * width))
    return "█" * filled + "░" * (width - filled)


# ── Terminal dashboard ────────────────────────────────────────────────────────

def print_monthly_dashboard(
    training_ledger: str = "carbonwatcher_ledger.json",
    llm_ledger: str = "carbonwatcher_llm_ledger.json",
    user_name: Optional[str] = None,
    organization: Optional[str] = None,
):
    data = combined_monthly(training_ledger, llm_ledger, user_name, organization)
    if not data:
        console.print("[yellow]No data found. Run some experiments first.[/yellow]")
        return

    title = "Monthly Carbon Emissions"
    if organization:
        title += f" — {organization}"
    if user_name:
        title += f" ({user_name})"

    # ── Summary table ──
    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Month",       style="bold")
    table.add_column("Train runs",  justify="right")
    table.add_column("LLM calls",   justify="right")
    table.add_column("Energy (Wh)", justify="right")
    table.add_column("Carbon (gCO2e)", justify="right")
    table.add_column("Cost (USD)",  justify="right")
    table.add_column("Trend",       no_wrap=True)

    months = sorted(data.keys())
    max_carbon = max((d["total_carbon_g"] for d in data.values()), default=1)

    prev_carbon = None
    for month in months:
        d = data[month]
        carbon = d["total_carbon_g"]

        # Trend arrow vs previous month
        if prev_carbon is None:
            trend = "—"
        elif carbon < prev_carbon:
            pct = (prev_carbon - carbon) / prev_carbon * 100
            trend = f"[green]↓ {pct:.1f}%[/green]"
        elif carbon > prev_carbon:
            pct = (carbon - prev_carbon) / prev_carbon * 100
            trend = f"[red]↑ {pct:.1f}%[/red]"
        else:
            trend = "→ 0%"
        prev_carbon = carbon

        table.add_row(
            month,
            str(d["training"].get("runs", 0)),
            str(d["llm"].get("calls", 0)),
            f"{d['total_energy_kwh']*1000:.3f}",
            f"{carbon:.4f}",
            f"${d['total_cost_usd']:.4f}",
            trend,
        )
    console.print(table)

    # ── ASCII bar chart ──
    console.print("\n[bold]Carbon emissions per month (gCO2e)[/bold]")
    for month in months:
        carbon = data[month]["total_carbon_g"]
        bar = _bar(carbon, max_carbon)
        color = "green" if carbon == min(d["total_carbon_g"] for d in data.values()) else (
                "red"   if carbon == max_carbon else "yellow")
        console.print(f"  {month}  [{color}]{bar}[/{color}]  {carbon:.4f} g")

    # ── Totals panel ──
    total_carbon = sum(d["total_carbon_g"] for d in data.values())
    total_cost   = sum(d["total_cost_usd"] for d in data.values())
    total_energy = sum(d["total_energy_kwh"] for d in data.values())
    console.print(Panel(
        f"[bold cyan]Total energy:[/bold cyan]  {total_energy*1000:.3f} Wh\n"
        f"[bold yellow]Total carbon:[/bold yellow]  {total_carbon:.4f} gCO2e"
        f"  ({total_carbon/1000:.6f} kg)\n"
        f"[bold green]Total cost:[/bold green]    ${total_cost:.4f} USD\n"
        f"[bold]Months tracked:[/bold] {len(months)}",
        title="[bold]All-time totals[/bold]", border_style="cyan",
    ))


# ── Export ────────────────────────────────────────────────────────────────────

def export_json(
    output_path: str,
    training_ledger: str = "carbonwatcher_ledger.json",
    llm_ledger: str = "carbonwatcher_llm_ledger.json",
    user_name: Optional[str] = None,
    organization: Optional[str] = None,
):
    data = combined_monthly(training_ledger, llm_ledger, user_name, organization)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user_name": user_name or "",
        "organization": organization or "",
        "months": data,
        "totals": {
            "total_carbon_g":   sum(d["total_carbon_g"]   for d in data.values()),
            "total_cost_usd":   sum(d["total_cost_usd"]   for d in data.values()),
            "total_energy_kwh": sum(d["total_energy_kwh"] for d in data.values()),
        },
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"[green]Exported JSON → {output_path}[/green]")


def export_csv(
    output_path: str,
    training_ledger: str = "carbonwatcher_ledger.json",
    llm_ledger: str = "carbonwatcher_llm_ledger.json",
    user_name: Optional[str] = None,
    organization: Optional[str] = None,
):
    data = combined_monthly(training_ledger, llm_ledger, user_name, organization)
    rows = []
    for month, d in sorted(data.items()):
        rows.append({
            "month":           month,
            "train_runs":      d["training"].get("runs", 0),
            "llm_calls":       d["llm"].get("calls", 0),
            "energy_kwh":      d["total_energy_kwh"],
            "carbon_g":        d["total_carbon_g"],
            "carbon_kg":       d["total_carbon_g"] / 1000,
            "cost_usd":        d["total_cost_usd"],
            "llm_tokens":      d["llm"].get("total_tokens", 0),
            "user_name":       user_name or "",
            "organization":    organization or "",
        })
    if not rows:
        console.print("[yellow]No data to export.[/yellow]")
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"[green]Exported CSV → {output_path}[/green]")
