"""Rich terminal report for LLM API usage tracking."""

from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from carbonwatcher.llm.tracker import LLMTracker, LLMCallRecord

console = Console()


def print_llm_report(tracker: LLMTracker, experiment: Optional[str] = None):
    records = (
        tracker.get_calls_by_experiment(experiment)
        if experiment
        else tracker.get_all_calls()
    )
    if not records:
        console.print("[yellow]No LLM calls recorded.[/yellow]")
        return

    title = f"LLM Experiment: {experiment}" if experiment else "All LLM Calls"
    console.print(f"\n[bold white]{title}[/bold white]  ({len(records)} calls)\n")

    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Call ID",    style="dim")
    table.add_column("Model",      style="cyan")
    table.add_column("In tok",     justify="right")
    table.add_column("Out tok",    justify="right")
    table.add_column("Energy (mWh)", justify="right")
    table.add_column("Carbon (µgCO2)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Latency",    justify="right")

    for r in records:
        table.add_row(
            r.call_id,
            r.model_id,
            f"{r.input_tokens:,}",
            f"{r.output_tokens:,}",
            f"{r.total_energy_wh*1000:.3f}",
            f"{r.carbon_g*1_000_000:.1f}",
            f"${r.total_cost_usd:.6f}",
            f"{r.latency_s:.2f}s" if r.latency_s else "—",
        )
    console.print(table)

    agg = tracker.aggregate(records)
    console.print(Panel(
        f"[bold]Calls:[/bold]          {agg['num_calls']}\n"
        f"[bold]Total tokens:[/bold]   {agg['total_tokens']:,}  "
        f"({agg['total_input_tokens']:,} in + {agg['total_output_tokens']:,} out)\n"
        f"[bold cyan]Total energy:[/bold cyan]   {agg['total_energy_wh']*1000:.4f} mWh\n"
        f"[bold yellow]Total carbon:[/bold yellow]   {agg['total_carbon_g']*1000:.4f} µgCO2e\n"
        f"[bold green]Total cost:[/bold green]     ${agg['total_cost_usd']:.6f} USD\n"
        f"[bold]Models used:[/bold]    {', '.join(agg['models_used'])}",
        title="[bold]LLM Usage Aggregate[/bold]",
        border_style="cyan",
    ))


def print_model_comparison(
    tracker: LLMTracker,
    input_tokens: int = 1000,
    output_tokens: int = 500,
):
    rows = tracker.compare_models(input_tokens=input_tokens, output_tokens=output_tokens)

    table = Table(
        title=f"Model Carbon Comparison  ({input_tokens:,} input + {output_tokens:,} output tokens)",
        box=box.SIMPLE_HEAVY, header_style="bold magenta",
    )
    table.add_column("Provider",    style="dim")
    table.add_column("Model",       style="cyan")
    table.add_column("gCO2/kWh",   justify="right")
    table.add_column("Carbon (µg)", justify="right")
    table.add_column("Energy (mWh)", justify="right")
    table.add_column("Cost (USD)",  justify="right")

    for i, row in enumerate(rows):
        color = "green" if i < 3 else ("yellow" if i < 8 else "red")
        table.add_row(
            row["provider"],
            f"[{color}]{row['model']}[/{color}]",
            f"{row['grid_gco2_kwh']:.0f}",
            f"{row['carbon_g']*1_000_000:.2f}",
            f"{row['energy_mwh']:.4f}",
            f"${row['cost_usd']:.6f}",
        )
    console.print(table)
    console.print("[dim]Green = lowest carbon  |  Carbon = energy × provider grid intensity[/dim]")
