"""Parse and aggregate CarbonTracker log files.

Usage:
    from carbonwatcher import parser

    logs = parser.parse_all_logs("./logs/")
    for log in logs:
        print(log["actual"]["carbon_g"], "gCO2e")

    parser.print_aggregate("./logs/")
"""

import os
import json
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def parse_log(path: str) -> dict:
    """Parse a single CarbonTracker .log JSON file. Returns the raw dict."""
    with open(path) as f:
        return json.load(f)


def parse_all_logs(log_dir: str, experiment: Optional[str] = None) -> List[dict]:
    """
    Parse all .log files in log_dir.
    If experiment is given, only return logs matching that experiment name.
    Returns list of dicts sorted by timestamp (oldest first).
    """
    if not os.path.isdir(log_dir):
        return []

    logs = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log"):
            continue
        path = os.path.join(log_dir, fname)
        try:
            entry = parse_log(path)
            if experiment and entry.get("experiment") != experiment:
                continue
            logs.append(entry)
        except (json.JSONDecodeError, KeyError):
            continue

    return sorted(logs, key=lambda x: x.get("timestamp", ""))


def aggregate_logs(logs: List[dict]) -> dict:
    """Compute aggregate stats from a list of parsed logs."""
    if not logs:
        return {}

    total_energy = sum(l["actual"]["energy_kwh"] for l in logs)
    total_carbon = sum(l["actual"]["carbon_g"] for l in logs)
    total_cost   = sum(l["actual"]["cost_usd"] for l in logs)
    total_dur    = sum(l["actual"]["duration_s"] for l in logs)

    return {
        "num_runs": len(logs),
        "total_duration_h": total_dur / 3600,
        "total_energy_kwh": total_energy,
        "total_carbon_g": total_carbon,
        "total_carbon_kg": total_carbon / 1000,
        "total_cost_usd": total_cost,
        "avg_energy_kwh": total_energy / len(logs),
        "avg_carbon_g": total_carbon / len(logs),
        "avg_cost_usd": total_cost / len(logs),
        "experiments": list({l.get("experiment", "?") for l in logs}),
        "models": list({l.get("model", "?") for l in logs}),
    }


def print_aggregate(log_dir: str, experiment: Optional[str] = None):
    """Print a Rich summary of all logs in log_dir to the terminal."""
    logs = parse_all_logs(log_dir, experiment=experiment)
    if not logs:
        console.print(f"[yellow]No logs found in {log_dir}[/yellow]")
        return

    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Run ID",     style="dim")
    table.add_column("Experiment")
    table.add_column("Model",      style="cyan")
    table.add_column("Duration",   justify="right")
    table.add_column("Energy (Wh)", justify="right")
    table.add_column("Carbon (gCO2e)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Region")

    for l in logs:
        a = l.get("actual", {})
        loc = l.get("location", {})
        table.add_row(
            l.get("run_id", "?"),
            l.get("experiment", "?"),
            l.get("model", "?"),
            f"{a.get('duration_s', 0):.1f}s",
            f"{a.get('energy_kwh', 0)*1000:.4f}",
            f"{a.get('carbon_g', 0):.6f}",
            f"${a.get('cost_usd', 0):.4f}",
            loc.get("region", "?"),
        )
    console.print(table)

    agg = aggregate_logs(logs)
    title = f"Aggregate — {experiment}" if experiment else "Aggregate — All Runs"
    console.print(Panel(
        f"[bold]Runs:[/bold]          {agg['num_runs']}\n"
        f"[bold]Total time:[/bold]    {agg['total_duration_h']:.3f} h\n"
        f"[bold cyan]Total energy:[/bold cyan]  {agg['total_energy_kwh']*1000:.4f} Wh\n"
        f"[bold yellow]Total carbon:[/bold yellow]  {agg['total_carbon_g']:.6f} gCO2e"
        f"  ({agg['total_carbon_kg']:.8f} kg)\n"
        f"[bold green]Total cost:[/bold green]    ${agg['total_cost_usd']:.4f} USD\n"
        f"[bold]Experiments:[/bold]   {', '.join(agg['experiments'])}\n"
        f"[bold]Models:[/bold]        {', '.join(agg['models'])}",
        title=f"[bold]{title}[/bold]",
        border_style="cyan",
    ))
