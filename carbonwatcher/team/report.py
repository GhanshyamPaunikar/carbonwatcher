"""Team carbon report — aggregate multiple ledger files, rank by person/org."""

import json
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


# ── Multi-ledger loading ──────────────────────────────────────────────────────

def load_team_ledgers(
    ledger_paths: List[str],
    llm_ledger_paths: Optional[List[str]] = None,
) -> tuple[List[RunRecord], List[LLMCallRecord]]:
    """Load records from multiple training and LLM ledger files."""
    all_train: List[RunRecord] = []
    for path in ledger_paths:
        if os.path.exists(path):
            ct = CostTracker(ledger_path=path)
            all_train.extend(ct.get_all_runs())

    all_llm: List[LLMCallRecord] = []
    for path in (llm_ledger_paths or []):
        if os.path.exists(path):
            lt = LLMTracker(ledger_path=path)
            all_llm.extend(lt.get_all_calls())

    return all_train, all_llm


# ── Per-user aggregation ──────────────────────────────────────────────────────

def _group_by_user(records: List[RunRecord]) -> dict[str, list]:
    by_user: dict[str, list] = defaultdict(list)
    for r in records:
        key = r.user_name or r.metadata.get("user_name", "unknown")
        by_user[key].append(r)
    return dict(by_user)


def _group_llm_by_experiment(records: List[LLMCallRecord]) -> dict[str, list]:
    by_exp: dict[str, list] = defaultdict(list)
    for r in records:
        by_exp[r.experiment].append(r)
    return dict(by_exp)


def team_stats(
    ledger_paths: List[str],
    llm_ledger_paths: Optional[List[str]] = None,
    month: Optional[str] = None,   # "YYYY-MM" filter
) -> dict[str, dict]:
    """
    Return per-user stats dict.
    Keys: user_name. Values: carbon, cost, energy, runs, tokens.
    """
    train_records, llm_records = load_team_ledgers(ledger_paths, llm_ledger_paths)

    # Optional month filter
    if month:
        train_records = [r for r in train_records if r.timestamp.startswith(month)]
        llm_records   = [r for r in llm_records   if r.timestamp.startswith(month)]

    by_user = _group_by_user(train_records)
    # Map LLM records to users by experiment prefix (best effort)
    llm_by_exp = _group_llm_by_experiment(llm_records)

    # Build per-user stats
    all_users = set(by_user.keys())
    result = {}
    for user in all_users:
        recs = by_user.get(user, [])
        # Try to match LLM records by user metadata
        user_llm = [r for r in llm_records
                    if r.metadata.get("user_name", "") == user]

        result[user] = {
            "user_name":    user,
            "organization": recs[0].organization if recs else "",
            "user_region":  recs[0].user_region  if recs else "",
            "train_runs":   len(recs),
            "train_carbon_g":  sum(r.carbon_kg * 1000 for r in recs),
            "train_cost_usd":  sum(r.total_cost_usd for r in recs),
            "train_energy_kwh": sum(r.energy_kwh for r in recs),
            "llm_calls":       len(user_llm),
            "llm_carbon_g":    sum(r.carbon_g for r in user_llm),
            "llm_cost_usd":    sum(r.total_cost_usd for r in user_llm),
            "llm_tokens":      sum(r.total_tokens for r in user_llm),
            "total_carbon_g":  sum(r.carbon_kg * 1000 for r in recs) + sum(r.carbon_g for r in user_llm),
            "total_cost_usd":  sum(r.total_cost_usd for r in recs) + sum(r.total_cost_usd for r in user_llm),
        }

    return dict(sorted(result.items(), key=lambda x: x[1]["total_carbon_g"], reverse=True))


# ── Terminal report ───────────────────────────────────────────────────────────

def print_team_report(
    ledger_paths: List[str],
    llm_ledger_paths: Optional[List[str]] = None,
    month: Optional[str] = None,
    organization: Optional[str] = None,
):
    stats = team_stats(ledger_paths, llm_ledger_paths, month=month)
    if organization:
        stats = {u: s for u, s in stats.items() if s["organization"] == organization}

    if not stats:
        console.print("[yellow]No team data found.[/yellow]")
        return

    title = "Team Carbon Report"
    if month:
        title += f" — {month}"
    if organization:
        title += f" — {organization}"

    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Rank",       justify="right", style="dim")
    table.add_column("Member",     style="bold")
    table.add_column("Region")
    table.add_column("Train runs", justify="right")
    table.add_column("LLM calls",  justify="right")
    table.add_column("LLM tokens", justify="right")
    table.add_column("Carbon (gCO2e)", justify="right")
    table.add_column("Cost (USD)", justify="right")

    max_carbon = max((s["total_carbon_g"] for s in stats.values()), default=1)

    for rank, (user, s) in enumerate(stats.items(), 1):
        carbon = s["total_carbon_g"]
        pct = carbon / max_carbon if max_carbon > 0 else 0
        color = "red" if pct > 0.8 else ("yellow" if pct > 0.4 else "green")
        table.add_row(
            str(rank),
            user,
            s["user_region"] or "—",
            str(s["train_runs"]),
            str(s["llm_calls"]),
            f"{s['llm_tokens']:,}",
            f"[{color}]{carbon:.4f}[/{color}]",
            f"${s['total_cost_usd']:.4f}",
        )
    console.print(table)

    total_carbon = sum(s["total_carbon_g"] for s in stats.values())
    total_cost   = sum(s["total_cost_usd"] for s in stats.values())
    top_user     = next(iter(stats))

    console.print(Panel(
        f"[bold]Members:[/bold]       {len(stats)}\n"
        f"[bold yellow]Total carbon:[/bold yellow]  {total_carbon:.4f} gCO2e\n"
        f"[bold green]Total cost:[/bold green]    ${total_cost:.4f} USD\n"
        f"[bold red]Top emitter:[/bold red]   {top_user} "
        f"({stats[top_user]['total_carbon_g']:.4f} gCO2e)",
        title="[bold]Team Summary[/bold]", border_style="cyan",
    ))
