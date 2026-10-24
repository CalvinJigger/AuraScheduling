"""Synthetic data generator for AI data center co-scheduling."""
import numpy as np
import pandas as pd


def generate_prices(T=24, seed=42):
    """Time-of-use electricity price ($/kWh).
    Peak: 10:00-12:00, 18:00-21:00; Valley: 23:00-7:00; Flat: rest."""
    rng = np.random.default_rng(seed)
    price = np.full(T, 0.60)  # flat baseline ¥/kWh ~ $0.085
    for h in range(T):
        if 10 <= h < 12 or 18 <= h < 21:
            price[h] = 1.20  # peak
        elif 23 <= h or h < 7:
            price[h] = 0.30  # valley
        elif 12 <= h < 18:
            price[h] = 0.85  # shoulder
    price += rng.normal(0, 0.03, T)
    return np.clip(price, 0.15, 2.0)


def generate_carbon(T=24, seed=43):
    """Grid carbon intensity (kgCO2/kWh).
    Higher at night (coal baseload), lower midday (solar displacing coal)."""
    rng = np.random.default_rng(seed)
    carbon = np.zeros(T)
    for h in range(T):
        # daytime (8-18): solar pushes carbon down
        if 8 <= h < 18:
            carbon[h] = 0.45 - 0.15 * np.exp(-((h - 13) ** 2) / 8)
        else:
            carbon[h] = 0.65  # night coal
    carbon += rng.normal(0, 0.02, T)
    return np.clip(carbon, 0.2, 0.9)


def generate_green(T=24, seed=44):
    """Green power supply available (kW): solar (day) + wind (night).
    Sized to cover ~40-50% of total energy on a typical day."""
    rng = np.random.default_rng(seed)
    solar = np.zeros(T)
    for h in range(T):
        if 6 <= h < 19:
            solar[h] = 600 * np.exp(-((h - 13) ** 2) / 18)
    # Wind: modest overnight contribution
    wind = 120 + 60 * np.sin(np.linspace(0, 2 * np.pi, T) + 1.5)
    wind = np.clip(wind, 30, 220)
    green = solar + wind + rng.normal(0, 20, T)
    return np.clip(green, 0, None)


def generate_inference_load(T=24, seed=45):
    """Fixed interactive inference load (kW) - cannot be shifted.
    Follows user activity: ramps up morning, peaks afternoon, drops at night."""
    rng = np.random.default_rng(seed)
    load = np.zeros(T)
    for h in range(T):
        base = 250
        if 9 <= h < 12 or 14 <= h < 18:
            base = 500  # business hours peak
        elif 12 <= h < 14:
            base = 400
        elif 20 <= h or h < 6:
            base = 180  # night
        load[h] = base
    load += rng.normal(0, 20, T)
    return np.clip(load, 100, None)


def generate_jobs(n_jobs=12, T=24, seed=46):
    """Deferrable batch jobs (training, ETL, fine-tuning).
    Each job: total energy E_i (kWh), release hour, deadline."""
    rng = np.random.default_rng(seed)
    jobs = []
    types = ["training", "finetune", "etl", "batch_infer"]
    for i in range(n_jobs):
        t = types[rng.integers(0, len(types))]
        # Energy: training big, etl small
        if t == "training":
            E = rng.uniform(800, 1500)
        elif t == "finetune":
            E = rng.uniform(300, 600)
        elif t == "batch_infer":
            E = rng.uniform(200, 400)
        else:
            E = rng.uniform(100, 250)
        release = int(rng.integers(0, T - 8))
        deadline = int(min(T - 1, release + rng.integers(6, 16)))
        jobs.append({
            "id": i,
            "type": t,
            "energy_kwh": E,
            "release": release,
            "deadline": deadline,
        })
    return pd.DataFrame(jobs)


def battery_params():
    """Hithium-style BESS: 2 MWh, 1 MW power, round-trip eff 0.9."""
    return {
        "capacity_kwh": 2000.0,
        "max_charge_kw": 1000.0,
        "max_discharge_kw": 1000.0,
        "eta_rt": 0.90,          # round-trip
        "eta_charge": np.sqrt(0.90),
        "eta_discharge": np.sqrt(0.90),
        "soc_init": 500.0,
        "soc_min": 200.0,
        "degradation_cost": 0.05,  # $ per kWh cycled
    }


def system_params():
    return {
        "gpu_total_kw": 1500.0,     # max GPU draw (deferrable + inference)
        "grid_limit_kw": 2000.0,     # contracted grid import limit
        "carbon_price": 50.0,        # $/tonCO2 = $0.05/kg
        "dt_hours": 1.0,
    }


def build_scenario(T=24, seed=42):
    return {
        "T": T,
        "price": generate_prices(T, seed),
        "carbon": generate_carbon(T, seed + 1),
        "green": generate_green(T, seed + 2),
        "inference": generate_inference_load(T, seed + 3),
        "jobs": generate_jobs(12, T, seed + 4),
        "battery": battery_params(),
        "sys": system_params(),
    }


if __name__ == "__main__":
    s = build_scenario()
    print("Price range:", s["price"].min().round(3), "-", s["price"].max().round(3))
    print("Carbon range:", s["carbon"].min().round(3), "-", s["carbon"].max().round(3))
    print("Green peak:", s["green"].max().round(0), "kW")
    print("Inference range:", s["inference"].min().round(0), "-", s["inference"].max().round(0), "kW")
    print("\nJobs:")
    print(s["jobs"])
