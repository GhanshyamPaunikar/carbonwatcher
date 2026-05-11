"""CLI for Carbon Trace.

Script-wrapping usage (no code changes needed):
    carbonwatcher run python train.py
    carbonwatcher run --region eu-west-1 --log-dir ./logs -- python train.py

Report/analysis commands:
    carbonwatcher runs
    carbonwatcher logs --log-dir ./logs
    carbonwatcher estimate --energy 0.01 --region eu-west-1
    carbonwatcher regions
    carbonwatcher llm-estimate -m claude-sonnet-4-6 -i 1000 -o 500
    carbonwatcher llm-compare
    carbonwatcher llm-models
"""

import sys
import subprocess
import click
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

from carbonwatcher.core.cost import CostTracker
from carbonwatcher.core.carbon import CarbonEstimator, GRID_INTENSITY

console = Console()

LEDGER_OPTION = click.option(
    "--ledger", default="carbonwatcher_ledger.json",
    show_default=True, help="Path to the run ledger JSON file."
)


# ── Root group ────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.3.0", prog_name="carbonwatcher")
def cli():
    """Carbon Trace — monitor energy, carbon, and cost of ML experiments.

    \b
    Wrap any script (no code changes required):
        carbonwatcher run python train.py
        carbonwatcher run --region eu-west-1 --log-dir ./logs -- python train.py
    """


# ── Script-wrapping command ───────────────────────────────────────────────────

@cli.command("run", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--log-dir",        default=None,       help="Directory to write log files.")
@click.option("--ledger",         default="carbonwatcher_ledger.json", show_default=True)
@click.option("--region",         default="us-east-1", show_default=True)
@click.option("--infrastructure", default="cloud",    show_default=True,
              type=click.Choice(["cloud", "on_premise", "laptop", "workstation"]))
@click.option("--epochs",         default=1,           show_default=True, type=int,
              help="Total epochs planned (used for prediction).")
@click.option("--monitor-epochs", default=1,           show_default=True, type=int)
@click.option("--verbose",        default=1,           show_default=True, type=int,
              help="0=silent, 1=summary, 2=debug.")
@click.argument("cmd", nargs=-1, required=True, type=click.UNPROCESSED)
def run_cmd(log_dir, ledger, region, infrastructure, epochs, monitor_epochs, verbose, cmd):
    """Track any script without modifying its code.

    \b
    Examples:
        carbonwatcher run python train.py
        carbonwatcher run --region eu-west-1 -- python train.py --epochs 50
    """
    from carbonwatcher.core.tracker import CarbonTracker

    console.print(f"[dim]Carbon Tracker: monitoring `{' '.join(cmd)}`[/dim]")
    tracker = CarbonTracker(
        epochs=epochs,
        monitor_epochs=monitor_epochs,
        experiment_name="cli",
        model_name=" ".join(cmd),
        region=region,
        infrastructure=infrastructure,
        ledger_path=ledger,
        log_dir=log_dir,
        verbose=verbose,
    )
    with tracker:
        proc = subprocess.run(list(cmd))
    sys.exit(proc.returncode)


# ── Training run commands ─────────────────────────────────────────────────────

@cli.command("runs")
@LEDGER_OPTION
@click.option("--experiment", "-e", default=None, help="Filter by experiment name.")
def list_runs(ledger, experiment):
    """List all recorded training runs from the ledger."""
    tracker = CostTracker(ledger_path=ledger)
    records = (
        tracker.get_runs_by_experiment(experiment) if experiment
        else tracker.get_all_runs()
    )
    if not records:
        console.print("[yellow]No runs recorded.[/yellow]")
        return

    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Run ID",   style="dim")
    table.add_column("Experiment")
    table.add_column("Model",    style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("Energy (Wh)", justify="right")
    table.add_column("Carbon (gCO2e)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Region")

    for r in records:
        table.add_row(
            r.run_id,
            r.experiment_name,
            r.model_name,
            f"{r.duration_s:.1f}s",
            f"{r.energy_kwh * 1000:.4f}",
            f"{r.carbon_kg * 1000:.4f}",
            f"${r.total_cost_usd:.4f}",
            r.region,
        )
    console.print(table)

    agg = tracker.aggregate(records)
    if agg:
        console.print(Panel(
            f"[bold]Runs:[/bold]          {agg['num_runs']}\n"
            f"[bold cyan]Total energy:[/bold cyan]  {agg['total_energy_kwh']*1000:.4f} Wh\n"
            f"[bold yellow]Total carbon:[/bold yellow]  {agg['total_carbon_kg']*1000:.4f} gCO2e\n"
            f"[bold green]Total cost:[/bold green]    ${agg['total_cost_usd']:.4f} USD",
            title="[bold]Aggregate[/bold]", border_style="cyan",
        ))


@cli.command("logs")
@click.option("--log-dir", default="./logs", show_default=True,
              help="Directory containing .log files.")
@click.option("--experiment", "-e", default=None, help="Filter by experiment name.")
def logs_cmd(log_dir, experiment):
    """Parse and display structured log files."""
    from carbonwatcher import parser
    parser.print_aggregate(log_dir, experiment=experiment)


@cli.command("compare")
@LEDGER_OPTION
@click.option("--metric", "-m",
              type=click.Choice(["carbon_per_dollar", "energy_per_dollar"]),
              default="carbon_per_dollar", show_default=True)
def compare(ledger, metric):
    """Rank runs by efficiency."""
    tracker = CostTracker(ledger_path=ledger)
    ranking = tracker.efficiency_ranking(metric=metric)
    if not ranking:
        console.print("[yellow]No runs with cost > 0 to rank.[/yellow]")
        return

    table = Table(title=f"Efficiency Ranking — {metric}", box=box.ROUNDED,
                  header_style="bold magenta")
    table.add_column("Rank",  justify="right", style="dim")
    table.add_column("Run ID")
    table.add_column("Experiment")
    table.add_column("Model",  style="cyan")
    table.add_column(metric.replace("_", " ").title(), justify="right")

    for i, row in enumerate(ranking, 1):
        table.add_row(str(i), row["run_id"], row["experiment"],
                      row["model"], f"{row['score']:.6f}")
    console.print(table)
    console.print("[green]Lower = more efficient.[/green]")


@cli.command("estimate")
@click.option("--energy", "-e", required=True, type=float,
              help="Energy in kWh.")
@click.option("--region", "-r", default="us-east-1", show_default=True)
def estimate(energy, region):
    """Quick carbon estimate for a given energy amount and region."""
    try:
        est = CarbonEstimator(region=region)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    r = est.estimate(energy_kwh=energy)
    console.print(
        f"\n[bold cyan]Energy:[/bold cyan]  {energy*1000:.4f} Wh\n"
        f"[bold]Region:[/bold]  {region} @ {r.grid_intensity_gco2_per_kwh} gCO2/kWh"
        f"  [{r.intensity_source}]\n"
        f"[bold yellow]Carbon:[/bold yellow]  {r.carbon_g:.6f} gCO2e  ({r.carbon_kg:.8f} kg)\n"
        f"[bold green]Equiv:[/bold green]   {r.equivalents['smartphone_charges']:.4f} phone charges"
        f"  |  {r.equivalents['km_driven']:.6f} km driven\n"
        f"[bold]Social cost:[/bold] ${r.social_cost_usd:.8f} USD\n"
    )


@cli.command("regions")
@click.option("--energy", "-e", default=0.001, show_default=True,
              help="Energy in kWh to compare across regions.")
def regions(energy):
    """Compare carbon footprint across all known regions."""
    table = Table(
        title=f"Region Comparison — {energy*1000:.3f} Wh",
        box=box.SIMPLE_HEAVY, header_style="bold magenta",
    )
    table.add_column("Region",  style="cyan")
    table.add_column("gCO2/kWh", justify="right")
    table.add_column("Carbon (mgCO2e)", justify="right")

    for region, intensity in sorted(GRID_INTENSITY.items(), key=lambda x: x[1]):
        carbon_mg = energy * intensity * 1000
        color = "green" if intensity < 200 else ("yellow" if intensity < 400 else "red")
        table.add_row(region, f"[{color}]{intensity:.0f}[/{color}]", f"{carbon_mg:.3f}")
    console.print(table)


# ── LLM commands ──────────────────────────────────────────────────────────────

LLM_LEDGER_OPTION = click.option(
    "--ledger", default="carbonwatcher_llm_ledger.json",
    show_default=True, help="Path to the LLM call ledger JSON file.",
)


@cli.command("llm-runs")
@LLM_LEDGER_OPTION
@click.option("--experiment", "-e", default=None)
def llm_runs(ledger, experiment):
    """List all recorded LLM API calls."""
    from carbonwatcher.llm.tracker import LLMTracker
    from carbonwatcher.llm.report import print_llm_report
    tracker = LLMTracker(ledger_path=ledger)
    print_llm_report(tracker, experiment=experiment)


@cli.command("llm-compare")
@click.option("--input-tokens",  "-i", default=1000, show_default=True)
@click.option("--output-tokens", "-o", default=500,  show_default=True)
@click.option("--provider", "-p", default=None,
              help="Filter by provider (anthropic, openai, google, mistral).")
def llm_compare(input_tokens, output_tokens, provider):
    """Compare carbon footprint of all known LLM models."""
    from carbonwatcher.llm.tracker import LLMTracker
    from carbonwatcher.llm.report import print_model_comparison
    tracker = LLMTracker(ledger_path="/dev/null")
    if provider:
        orig = tracker.compare_models
        tracker.compare_models = lambda input_tokens, output_tokens: [
            r for r in orig(input_tokens, output_tokens)
            if r["provider"] == provider.lower()
        ]
    print_model_comparison(tracker, input_tokens=input_tokens, output_tokens=output_tokens)


@cli.command("llm-estimate")
@click.option("--model",         "-m", required=True,
              help="Model ID (e.g. claude-sonnet-4-6, gpt-4o).")
@click.option("--input-tokens",  "-i", default=1000, show_default=True)
@click.option("--output-tokens", "-o", default=500,  show_default=True)
def llm_estimate(model, input_tokens, output_tokens):
    """Estimate energy, carbon, and cost for a single LLM API call."""
    from carbonwatcher.llm.tracker import LLMTracker
    tracker = LLMTracker(ledger_path="/dev/null")
    result = tracker.record(model=model, input_tokens=input_tokens,
                            output_tokens=output_tokens)
    console.print("\n" + result.summary() + "\n")


@cli.command("llm-models")
@click.option("--provider", "-p", default=None,
              help="Filter by provider (anthropic, openai, google, mistral).")
def llm_models(provider):
    """List all known LLM models and their energy/carbon specs."""
    from carbonwatcher.llm.models import list_models
    models = list_models(provider=provider)
    table = Table(
        title="Known LLM Models — Energy & Carbon Specs",
        box=box.ROUNDED, header_style="bold magenta",
    )
    table.add_column("Provider",  style="dim")
    table.add_column("Model",     style="cyan")
    table.add_column("Wh/1k in",  justify="right")
    table.add_column("Wh/1k out", justify="right")
    table.add_column("Grid gCO2/kWh", justify="right")
    table.add_column("$/1M in",   justify="right")
    table.add_column("$/1M out",  justify="right")
    for m in models:
        table.add_row(
            m.provider, m.display_name,
            f"{m.wh_per_1k_input_tokens:.5f}",
            f"{m.wh_per_1k_output_tokens:.5f}",
            f"{m.provider_grid_intensity_gco2_kwh:.0f}",
            f"${m.cost_per_1m_input_tokens:.3f}",
            f"${m.cost_per_1m_output_tokens:.3f}",
        )
    console.print(table)


# ── Config command ────────────────────────────────────────────────────────────

@cli.command("config")
@click.option("--region",       default=None, help="Your local grid region (e.g. India, eu-west-1).")
@click.option("--user-name",    default=None, help="Your name (for team reports).")
@click.option("--organization", default=None, help="Your org/team name.")
@click.option("--show",   is_flag=True, help="Show current config.")
def config_cmd(region, user_name, organization, show):
    """Set your location and identity for accurate carbon tracking.

    \b
    Examples:
        carbonwatcher config --region India --user-name Alice --organization Acme
        carbonwatcher config --show
    """
    from carbonwatcher.core.location import load_config, save_config, detect_location, GRID_INTENSITY

    if show:
        cfg = load_config()
        if not cfg:
            console.print("[yellow]No config set yet.[/yellow]")
            detected, src = detect_location()
            console.print(f"Auto-detected region: [cyan]{detected}[/cyan] (via {src})")
            return
        for k, v in cfg.items():
            console.print(f"  [bold]{k}:[/bold] {v}")
        return

    updates = {}
    if region:
        if region not in GRID_INTENSITY:
            console.print(f"[red]Unknown region '{region}'.[/red] "
                          f"Run [bold]carbonwatcher regions[/bold] to see valid options.")
            raise SystemExit(1)
        updates["user_region"] = region
    if user_name:
        updates["user_name"] = user_name
    if organization:
        updates["organization"] = organization

    if not updates:
        console.print("Nothing to update. Use --region, --user-name, or --organization.")
        return

    save_config(updates)
    console.print("[green]Config saved.[/green]")
    for k, v in updates.items():
        console.print(f"  {k} = {v}")


# ── Dashboard commands ────────────────────────────────────────────────────────

@cli.command("dashboard")
@LEDGER_OPTION
@click.option("--llm-ledger",   default="carbonwatcher_llm_ledger.json", show_default=True)
@click.option("--user",         default=None, help="Filter by user name.")
@click.option("--organization", default=None, help="Filter by organization.")
@click.option("--export-json",  default=None, metavar="FILE", help="Export to JSON file.")
@click.option("--export-csv",   default=None, metavar="FILE", help="Export to CSV file.")
def dashboard(ledger, llm_ledger, user, organization, export_json, export_csv):
    """Monthly emissions dashboard with ASCII bar charts.

    \b
    Examples:
        carbonwatcher dashboard
        carbonwatcher dashboard --user Alice
        carbonwatcher dashboard --export-csv report.csv
    """
    from carbonwatcher.dashboard.monthly import (
        print_monthly_dashboard, export_json as _ejson, export_csv as _ecsv
    )
    from carbonwatcher.core.location import get_config_value

    _user = user or get_config_value("user_name")
    _org  = organization or get_config_value("organization")

    print_monthly_dashboard(
        training_ledger=ledger,
        llm_ledger=llm_ledger,
        user_name=_user,
        organization=_org,
    )
    if export_json:
        _ejson(export_json, ledger, llm_ledger, _user, _org)
    if export_csv:
        _ecsv(export_csv, ledger, llm_ledger, _user, _org)


# ── Goals commands ────────────────────────────────────────────────────────────

@cli.group("goals")
def goals_group():
    """Set and track carbon reduction goals."""


@goals_group.command("set")
@click.option("--monthly",  type=float, default=None, help="Monthly carbon goal in gCO2e.")
@click.option("--yearly",   type=float, default=None, help="Yearly carbon goal in gCO2e.")
@click.option("--cost-monthly", type=float, default=None, help="Monthly cost cap in USD.")
@click.option("--cost-yearly",  type=float, default=None, help="Yearly cost cap in USD.")
@click.option("--note",     default="", help="Optional note.")
def goals_set(monthly, yearly, cost_monthly, cost_yearly, note):
    """Set monthly or yearly carbon reduction targets.

    \b
    Examples:
        carbonwatcher goals set --monthly 500
        carbonwatcher goals set --yearly 3000 --cost-yearly 100
    """
    from carbonwatcher.dashboard.goals import set_goal
    if monthly is not None:
        set_goal("monthly", monthly, cost_monthly, note)
    if yearly is not None:
        set_goal("yearly", yearly, cost_yearly, note)
    if monthly is None and yearly is None:
        console.print("Specify --monthly or --yearly.")


@goals_group.command("progress")
@LEDGER_OPTION
@click.option("--llm-ledger", default="carbonwatcher_llm_ledger.json", show_default=True)
@click.option("--user",       default=None)
def goals_progress(ledger, llm_ledger, user):
    """Show progress toward your reduction goals."""
    from carbonwatcher.dashboard.goals import print_goals_progress
    from carbonwatcher.core.location import get_config_value
    _user = user or get_config_value("user_name")
    print_goals_progress(ledger, llm_ledger, _user)


# ── Team commands ─────────────────────────────────────────────────────────────

@cli.command("team")
@click.argument("ledgers", nargs=-1, required=True,
                metavar="LEDGER [LEDGER ...]")
@click.option("--llm-ledger",   multiple=True, metavar="FILE",
              help="LLM ledger file(s). Can repeat.")
@click.option("--month",        default=None, help="Filter to a month: YYYY-MM.")
@click.option("--organization", default=None, help="Filter by organization name.")
def team(ledgers, llm_ledger, month, organization):
    """Compare carbon emissions across team members.

    \b
    Point to each person's ledger file:
        carbonwatcher team alice.json bob.json carol.json
        carbonwatcher team /shared/ledgers/*.json --month 2025-04
    """
    from carbonwatcher.team.report import print_team_report
    print_team_report(
        ledger_paths=list(ledgers),
        llm_ledger_paths=list(llm_ledger) or None,
        month=month,
        organization=organization,
    )


if __name__ == "__main__":
    cli()
