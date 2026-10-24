# Aura Scheduling

**AI Data Center Power–Computing Co-Scheduling**
HackSHSID 2026 · Track 2 · Huawei × Xiamen Hithium Energy Storage

---

## What this does

AI data centers run 24/7 and electricity is one of their biggest costs. But electricity price, carbon intensity, and green power supply all change through the day. Some tasks (like a chat reply) must run instantly, but others (like model training or data cleaning) can wait a few hours.

This project builds a scheduler that decides, hour by hour:

1. Which flexible computing tasks to run
2. How much power to buy from the grid
3. When to charge and discharge the battery

The goal: cut daily electricity cost and carbon emissions without missing any task deadline.

## Results (10 random test scenarios)

| Method | Daily cost | Daily carbon | Green share | Missed deadlines |
|---|---|---|---|---|
| Run everything immediately (baseline) | $3,800 | 3,240 kg | 50% | 0 |
| Only run during cheap hours | $3,392 | 3,615 kg | 50% | 0 |
| Our heuristic scheduler | $2,854 | 3,706 kg | 50% | 0 |
| **MILP optimal (our method)** | **$1,853** | **3,667 kg** | **60%** | **0** |

Our optimal scheduler saves about **51% on daily cost** compared to the naive approach, raises green energy share from 50% to 60%, and never misses a deadline.

We also sweep a "carbon weight" dial from 0 to 50 to show the trade-off curve between cost and carbon — the operator can choose where on that curve they want to sit.

## How it works

The core model is a **Mixed-Integer Linear Program (MILP)** solved with CBC (via PuLP):

- **Decision variables**: task energy per hour, grid purchase, battery charge/discharge, battery state of charge
- **Objective**: minimize `electricity cost + carbon cost + battery degradation`
- **Constraints**: power balance, GPU capacity, task deadlines, battery SOC limits, round-trip efficiency, grid import limit

We also implement two baselines (AFAP = run immediately, TOU = only cheap hours) and a fast greedy heuristic for comparison.

## Project structure

```
├── data.py             # Synthetic data generator (prices, carbon, solar/wind, workloads, battery)
├── schedulers.py       # MILP optimizer + AFAP/TOU baselines + heuristic + evaluation
├── run_experiment.py   # Runs multi-seed experiment + Pareto sweep, builds dashboard.html
├── dashboard.html      # Visual report (open in browser)
├── cover.png           # Project cover image
└── requirements.txt    # Python dependencies
```

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## How to run

```bash
python run_experiment.py
```

This runs 10 random scenarios, sweeps the carbon weight, and writes `dashboard.html`. Open that file in any browser to see the charts.

To tweak the scenario (prices, battery size, number of tasks, etc.), edit the parameters at the bottom of `data.py`.

## Data sources and assumptions

**No real data was provided by the problem statement**, so we generated realistic synthetic data based on industry reference values:

| Parameter | Value used | Reference |
|---|---|---|
| Peak / flat / valley price | 1.20 / 0.85 / 0.30 ¥/kWh | Typical Chinese industrial TOU tariff |
| Grid carbon intensity | 0.3–0.7 kgCO₂/kWh | China grid average ~0.55 kg/kWh; lower midday due to solar |
| Solar output | 600 kW peak at noon, bell curve | Typical 1 MW on-site solar array |
| Wind output | 30–220 kW, overnight-biased | Modest on-site wind contribution |
| Battery size | 2 MWh / 1 MW / 90% round-trip | Hithium industrial BESS typical specs |
| Carbon price | $50/tonCO₂ | Used as a tunable weight (swept 0–50) |
| Inference load | 180–515 kW, business-hours peak | Fixed baseline traffic that cannot be shifted |
| Number of deferrable tasks | 12 per day (training, fine-tuning, ETL, batch inference) | Synthetic workload mix |

All assumptions are listed explicitly. If real data becomes available, replace the functions in `data.py` — the scheduler code does not need to change.

## Key findings

- **Cheap ≠ clean.** Running tasks during night-time valley hours saves money but often burns more carbon, because night grid power comes from coal baseload. This is why we make cost and carbon separate terms in the objective, not a single weighted guess.
- **Battery matters.** The MILP uses the battery to store midday solar surplus and discharge during evening peak, avoiding expensive grid purchase.
- **There is no single "right" answer.** The Pareto front lets the operator choose: pay 3× more to cut carbon in half, or save money and accept higher emissions.

## AI usage

We used Doubao as a coding assistant to write boilerplate Python, debug package errors, and set up the experiment loop. The problem framing, optimization model design, data assumptions, choice of baselines, and trade-off analysis were done by us. AI did not make any design decisions.

## License

MIT
