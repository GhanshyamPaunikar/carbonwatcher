# CarbonWatcher

**Track the energy, carbon footprint, and cost of every ML training run and LLM API call — automatically, with zero changes to your existing code.**

```bash
pip install carbonwatcher
```

[![PyPI version](https://img.shields.io/pypi/v/carbonwatcher)](https://pypi.org/project/carbonwatcher/)
[![Python](https://img.shields.io/pypi/pyversions/carbonwatcher)](https://pypi.org/project/carbonwatcher/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why CarbonWatcher?

AI is one of the fastest-growing sources of electricity demand on the planet. A single large model training run can emit as much CO₂ as five cars over their entire lifetime. Yet most ML teams have no idea what their compute actually costs the environment — or how much the carbon footprint varies depending on *where* that compute runs.

**CarbonWatcher makes the invisible visible:**

- Your training job running in India emits **8× more CO₂** than the same job in Ireland
- An API call to an LLM provider uses *their* grid, not yours — and CarbonWatcher tracks both separately
- Monthly dashboards, team leaderboards, and reduction goals give you the data to act on

No manual measurement. No configuration required to get started. Wrap your code or CLI and it just works.

---

## Features

- **Zero-code script wrapping** — run any Python script under CarbonWatcher with one command
- **Context manager & decorator API** — drop into existing training loops in two lines
- **Epoch-based tracking** — per-epoch stats and automatic extrapolation for long runs
- **Location-aware carbon** — auto-detects your region; uses provider grid for cloud LLM calls
- **LLM token tracking** — Anthropic, OpenAI, Ollama, HuggingFace, llama.cpp — token counts from API responses, no guessing
- **Hardware profiling** — Intel RAPL (direct energy), Apple Silicon powermetrics, NVIDIA NVML GPU, DRAM estimation
- **Monthly dashboard** — ASCII bar charts, trend arrows, JSON/CSV export
- **Reduction goals** — set monthly/yearly carbon budgets, track progress with visual progress bars
- **Team reports** — aggregate multiple ledgers, rank members by emissions
- **JSON ledger** — all runs persisted locally; queryable, exportable, shareable

---

## Installation

```bash
# Core — training tracking + CLI
pip install carbonwatcher

# With Anthropic API support
pip install "carbonwatcher[anthropic]"

# With OpenAI support
pip install "carbonwatcher[openai]"

# With local model support (Ollama, HuggingFace, llama.cpp)
pip install "carbonwatcher[ollama,huggingface,llamacpp]"

# Everything
pip install "carbonwatcher[all]"
```

**Requirements:** Python 3.9+, macOS or Linux

---

## Quick Start

### 1. Wrap any script — no code changes

```bash
carbonwatcher run python train.py
carbonwatcher run --region eu-west-1 --log-dir ./logs -- python train.py --epochs 50
```

### 2. Context manager

```python
from carbonwatcher import CarbonTracker

with CarbonTracker(experiment_name="resnet50", model_name="ResNet50") as ct:
    train()
```

Output:
```
Run:        a3f2c1b8
Experiment: resnet50  |  Model: ResNet50
CPU:        Intel Xeon  [rapl]
GPU:        NVIDIA A100 × 2
Duration:   142.3s
Energy:     3.1200 Wh  (CPU: 2.8400  GPU: 0.2240  DRAM: 0.0560)  PUE×1.10
Carbon:     2.2090 gCO2e  (0.00000221 kg)
Grid:       eu-west-1 @ 88.0 gCO2/kWh  [static_table]
Cost:       $0.0312 USD
Equiv:      0.0002 phone charges  |  0.000014 km driven
```

### 3. Epoch-based API (recommended for long training runs)

```python
from carbonwatcher import CarbonTracker

tracker = CarbonTracker(epochs=100, monitor_epochs=2, experiment_name="bert-finetune")
for epoch in range(100):
    tracker.epoch_start()
    train_one_epoch()
    tracker.epoch_end()   # prints per-epoch stats; after epoch 2 prints full prediction
tracker.stop()
```

After the monitored epochs, it extrapolates and prints:
```
  [Predicted for 100 epochs]
    Time:    01:24:00
    Energy:  840.0000 Wh
    Carbon:  168.0000 gCO2e
    Equiv:   0.000112 km driven
```

### 4. Decorator

```python
from carbonwatcher import CarbonTracker

@CarbonTracker.track(experiment="gpt2-finetune", model="GPT-2", region="us-west-2")
def train():
    ...
```

---

## LLM Token Tracking

Track the carbon and cost of every LLM API call. Token counts are read directly from API responses — no manual counting, no estimates.

### Anthropic

```python
from carbonwatcher.llm.anthropic_wrapper import TrackedAnthropic

client = TrackedAnthropic()  # drop-in replacement for anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain gradient descent."}],
)
# carbon + cost logged automatically from response usage metadata
```

Streaming:
```python
with client.messages.stream(model="claude-sonnet-4-6", max_tokens=512,
                             messages=[{"role": "user", "content": "Hello"}]) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
# tracked automatically on stream completion
```

### OpenAI

```python
from carbonwatcher.llm.openai_wrapper import TrackedOpenAI

client = TrackedOpenAI()  # drop-in replacement for openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarise this paper."}],
)
```

### Ollama (local models)

```python
from carbonwatcher.llm.ollama_wrapper import TrackedOllama

client = TrackedOllama()  # drop-in replacement for ollama.Client()
response = client.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is backpropagation?"}]
)
# token counts from Ollama's prompt_eval_count / eval_count fields
```

### HuggingFace (local models)

```python
from carbonwatcher.llm.huggingface_wrapper import TrackedHuggingFace

model = TrackedHuggingFace("mistralai/Mistral-7B-v0.1")
output = model.generate("The attention mechanism works by")
# exact token counts via AutoTokenizer
```

### llama.cpp

```python
from carbonwatcher.llm.llamacpp_wrapper import TrackedLlamaCpp

llm = TrackedLlamaCpp("models/llama-3.2-3b.gguf")
response = llm.chat([{"role": "user", "content": "Explain LoRA fine-tuning."}])
# token counts from llama.cpp usage.prompt_tokens / usage.completion_tokens
```

### Direct LLM tracking (any provider)

```python
from carbonwatcher.llm.tracker import LLMTracker

tracker = LLMTracker()
result = tracker.record(
    model="gpt-4o",
    input_tokens=1500,
    output_tokens=300,
    experiment="summarisation-pipeline",
    metadata={"task": "summarisation"},
)
print(f"Carbon: {result.carbon_g:.4f} gCO2e  Cost: ${result.total_cost_usd:.4f}")
```

---

## Location-Aware Carbon

Carbon intensity varies dramatically by region. CarbonWatcher auto-detects your location via IP geolocation and applies the correct grid intensity for your local compute. For cloud LLM API calls, the **provider's** grid is used — not yours.

| Region | Grid Intensity |
|--------|---------------|
| Norway | 28 gCO2/kWh |
| France | 58 gCO2/kWh |
| Ireland / eu-west-1 | 88 gCO2/kWh |
| UK | 233 gCO2/kWh |
| California / us-west-1 | 220 gCO2/kWh |
| USA avg / us-east-1 | 386 gCO2/kWh |
| India | 708 gCO2/kWh |
| Australia | 790 gCO2/kWh |

**Cloud LLM provider grids (built-in):**

| Provider | Grid Used |
|----------|-----------|
| Anthropic | ~200 gCO2/kWh (AWS + GCP blended) |
| OpenAI | ~200 gCO2/kWh |
| Mistral AI | 58 gCO2/kWh (French grid) |
| Local (Ollama, llama.cpp) | Your detected region |

Set your region permanently:
```bash
carbonwatcher config --region India --user-name Alice --organization Acme
carbonwatcher config --show
```

---

## CLI Reference

```
carbonwatcher --help
```

| Command | What it does |
|---------|-------------|
| `carbonwatcher run` | Wrap any script — `carbonwatcher run python train.py` |
| `carbonwatcher runs` | List all recorded training runs |
| `carbonwatcher compare` | Rank runs by carbon efficiency |
| `carbonwatcher estimate` | Quick carbon estimate for energy + region |
| `carbonwatcher regions` | Compare carbon intensity across all known regions |
| `carbonwatcher config` | Set your region, name, and organisation |
| `carbonwatcher dashboard` | Monthly emissions dashboard with ASCII bar charts |
| `carbonwatcher goals set` | Set monthly/yearly carbon targets |
| `carbonwatcher goals progress` | Show progress toward targets |
| `carbonwatcher team` | Compare emissions across team members |
| `carbonwatcher llm-runs` | List all recorded LLM API calls |
| `carbonwatcher llm-estimate` | Estimate carbon for one LLM call |
| `carbonwatcher llm-compare` | Rank all known models by carbon footprint |
| `carbonwatcher llm-models` | List all supported LLM models |
| `carbonwatcher logs` | Parse and display structured log files |

### Examples

```bash
# Track any script
carbonwatcher run python train.py
carbonwatcher run --region eu-west-1 --infrastructure on_premise -- python train.py

# View your runs
carbonwatcher runs
carbonwatcher runs --experiment bert-finetune

# Carbon estimate
carbonwatcher estimate --energy 0.5 --region India

# Compare regions
carbonwatcher regions

# Monthly dashboard
carbonwatcher dashboard
carbonwatcher dashboard --export-json report.json
carbonwatcher dashboard --export-csv report.csv

# Reduction goals
carbonwatcher goals set --monthly 500 --cost-monthly 10.0
carbonwatcher goals set --yearly 5000
carbonwatcher goals progress

# Team comparison (shared ledger files)
carbonwatcher team alice.json bob.json carol.json
carbonwatcher team /shared/ledgers/*.json --month 2026-05
carbonwatcher team alice.json --llm-ledger alice_llm.json

# LLM usage
carbonwatcher llm-runs
carbonwatcher llm-estimate -m gpt-4o -i 2000 -o 500
carbonwatcher llm-compare
```

---

## Monthly Dashboard

```bash
carbonwatcher dashboard
```

```
              Monthly Carbon Emissions — Acme (Alice)
╭─────────┬───────────┬───────────┬───────────┬────────────┬───────────┬────────╮
│ Month   │ Train runs│ LLM calls │ Energy(Wh)│Carbon(gCO2)│  Cost(USD)│  Trend │
├─────────┼───────────┼───────────┼───────────┼────────────┼───────────┼────────┤
│ 2026-03 │        12 │        84 │    24.310 │    17.2319 │   $0.8431 │      — │
│ 2026-04 │         9 │       102 │    18.940 │    13.4205 │   $0.6812 │ ↓22.1% │
│ 2026-05 │        15 │        91 │    29.110 │    20.6219 │   $1.0240 │ ↑53.7% │
╰─────────┴───────────┴───────────┴───────────┴────────────┴───────────┴────────╯

Carbon emissions per month (gCO2e)
  2026-03  ████████████████████░░░░░░░░░░  17.2319 g
  2026-04  ████████████████░░░░░░░░░░░░░░  13.4205 g
  2026-05  ██████████████████████████████  20.6219 g
```

---

## Reduction Goals

```bash
carbonwatcher goals set --monthly 500 --cost-monthly 10.0
carbonwatcher goals progress
```

```
                       Reduction Goals Progress
╭────────────────┬────────┬─────────────┬──────────────┬──────────────┬────────╮
│ Period         │ Metric │ Goal        │ Used         │ Remaining    │ Status │
├────────────────┼────────┼─────────────┼──────────────┼──────────────┼────────┤
│ 2026-05        │ Carbon │ 500.00 gCO2e│ 20.62 gCO2e  │ 479.38 gCO2e │ ✓ 4.1% │
│ (monthly)      │ Cost   │ 10.00 USD   │ 1.02 USD     │ 8.98 USD     │ ✓ 10.2%│
│ 2026 (yearly)  │ Carbon │ 5000.0 gCO2e│ 51.28 gCO2e  │ 4948.7 gCO2e │ ✓ 1.0% │
╰────────────────┴────────┴─────────────┴──────────────┴──────────────┴────────╯
```

---

## Team Reports

Each team member writes to their own ledger. The `team` command aggregates and ranks everyone by total emissions.

```bash
carbonwatcher team alice.json bob.json carol.json --month 2026-05
```

```
              Team Carbon Report — 2026-05
╭──────┬─────────┬──────────┬────────────┬───────────┬─────────────┬───────────╮
│ Rank │ Member  │ Region   │ Train runs │ LLM calls │Carbon(gCO2) │  Cost(USD)│
├──────┼─────────┼──────────┼────────────┼───────────┼─────────────┼───────────┤
│    1 │ alice   │ India    │         15 │        91 │     20.6219 │    $1.0240│
│    2 │ carol   │ us-east-1│         11 │        78 │      7.9823 │    $0.5910│
│    3 │ bob     │ eu-west-1│          9 │       102 │      1.8341 │    $0.4320│
╰──────┴─────────┴──────────┴────────────┴───────────┴─────────────┴───────────╯
  Members:       3
  Total carbon:  30.4383 gCO2e
  Top emitter:   alice (20.6219 gCO2e)
```

Alice emits **11× more** than Bob for similar workloads — same job, different grid. Switching to `eu-west-1` would cut her footprint by 87%.

---

## Python API

```python
from carbonwatcher import CarbonTracker
from carbonwatcher.core.cost import CostTracker
from carbonwatcher.llm.tracker import LLMTracker
from carbonwatcher.dashboard.monthly import combined_monthly, export_csv
from carbonwatcher.dashboard.goals import set_goal, print_goals_progress
from carbonwatcher.team.report import print_team_report
from carbonwatcher.parser import parse_all_logs, aggregate_logs

# Monthly data
data = combined_monthly("carbonwatcher_ledger.json", "carbonwatcher_llm_ledger.json")

# Goals
set_goal("monthly", carbon_g=500, cost_usd=10.0)
print_goals_progress()

# Team
print_team_report(["alice.json", "bob.json"], month="2026-05")

# Log files
logs = parse_all_logs("./logs", experiment="bert-finetune")
agg  = aggregate_logs(logs)
print(f"Total: {agg['total_carbon_g']:.4f} gCO2e over {agg['num_runs']} runs")
```

---

## Hardware Power Measurement

| Platform | Method | Accuracy |
|----------|--------|----------|
| Intel CPU (Linux) | RAPL — direct hardware energy counter | Exact |
| Apple Silicon (macOS) | `powermetrics` subprocess | High |
| NVIDIA GPU | NVML — direct power sensor | Exact |
| DRAM | RAM GB × 0.375 W/GB empirical ratio | Estimated |
| Other CPUs | TDP-based estimate | Conservative |

PUE applied per infrastructure:

| Infrastructure | PUE |
|----------------|-----|
| Cloud (AWS / GCP / Azure) | 1.10 |
| On-premise data centre | 1.50 |
| Workstation / Laptop | 1.00 |

---

## Use Cases

### Individual developer
- Know the carbon cost of each experiment before it becomes a habit
- Compare two approaches: does faster convergence also mean less carbon?
- See that switching from `us-east-1` to `eu-west-1` cuts training carbon by 77%

### ML research team
- Shared ledger → team leaderboard showing who drives emissions
- Set a monthly carbon budget per project
- Export monthly CSV for sustainability reporting

### Company / ESG reporting
- Scope 3 emissions data for AI compute — increasingly required in ESG disclosures
- Track year-over-year reduction vs targets
- Identify which team, model, or experiment is the top emitter

### Model selection
- `carbonwatcher llm-compare` ranks every supported model by carbon per 1K tokens
- A smaller, quantised local model often beats a cloud API on both cost and carbon
- Choose the right model for the right task with data, not guesswork

---

## Supported LLM Models

Run `carbonwatcher llm-models` to see the full list. A few highlights:

| Model | Provider | Approx. gCO2e / 1K tokens |
|-------|----------|-----------------------------|
| GPT-4o | OpenAI | ~0.0023 |
| GPT-4o mini | OpenAI | ~0.0003 |
| Sonnet (Anthropic) | Anthropic | ~0.0018 |
| Haiku (Anthropic) | Anthropic | ~0.0004 |
| Mistral Large | Mistral AI | ~0.0009 |
| Llama 3.2 3B (local) | Ollama | ~0.0001 |
| Mistral 7B (local) | Ollama | ~0.0002 |

Local models on efficient hardware in a low-carbon region can be **10–20× greener** than cloud APIs.

---

## Project Structure

```
carbonwatcher/
├── core/
│   ├── tracker.py       # CarbonTracker — main API
│   ├── hardware.py      # HardwareProfiler — RAPL, powermetrics, NVML
│   ├── energy.py        # EnergyCalculator — PUE, component breakdown
│   ├── carbon.py        # CarbonEstimator — grid intensity, equivalents
│   ├── cost.py          # CostTracker — ledger persistence
│   ├── grid.py          # GridIntensityResolver — static + live lookup
│   └── location.py      # Auto-detect region, config file
├── llm/
│   ├── tracker.py       # LLMTracker — token-based carbon/cost
│   ├── models.py        # ModelSpec database — all supported models
│   ├── tokenizer.py     # Token counting — HF, tiktoken, fallback
│   ├── anthropic_wrapper.py
│   ├── openai_wrapper.py
│   ├── ollama_wrapper.py
│   ├── huggingface_wrapper.py
│   └── llamacpp_wrapper.py
├── dashboard/
│   ├── monthly.py       # Monthly aggregation, ASCII charts, export
│   └── goals.py         # Reduction goals, progress tracking
├── team/
│   └── report.py        # Multi-ledger team aggregation and ranking
├── cli/
│   └── main.py          # Click CLI — all commands
└── parser.py            # Structured log file parser
```

---

## Contributing

Pull requests welcome. To set up a development environment:

```bash
git clone https://github.com/ghanshyampaunikar/carbonwatcher
cd carbonwatcher
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT — see [LICENSE](LICENSE)
