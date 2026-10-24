"""Schedulers: MILP optimal, AFAP baseline, TOU baseline, greedy heuristic."""
import numpy as np
import pulp


def _empty_result(T, n_jobs):
    return {
        "job_schedule": np.zeros((n_jobs, T)),   # kWh per job per hour
        "grid": np.zeros(T),                      # kW purchased
        "green_use": np.zeros(T),                 # kW green used
        "green_curtail": np.zeros(T),             # kW green curtailed
        "charge": np.zeros(T),                    # kW battery charge
        "discharge": np.zeros(T),                 # kW battery discharge
        "soc": np.zeros(T),                       # kWh battery SOC
        "total_load": np.zeros(T),                # kW total demand
    }


def evaluate(result, scenario, carbon_weight=1.0):
    """Compute metrics from a schedule result."""
    price = scenario["price"]
    carbon = scenario["carbon"]
    sys = scenario["sys"]
    bat = scenario["battery"]
    dt = sys["dt_hours"]

    cost = float(np.sum(price * result["grid"] * dt))
    carbon_kg = float(np.sum(carbon * result["grid"] * dt))
    deg = float(np.sum((result["charge"] + result["discharge"]) * bat["degradation_cost"] * dt))
    total = cost + carbon_weight * sys["carbon_price"] * carbon_kg / 1000.0 * 1000.0 + deg
    # carbon_price is $/tonCO2; carbon_kg is kg, so divide by 1000 to get tons
    total = cost + carbon_weight * sys["carbon_price"] * carbon_kg / 1000.0 + deg

    # SLA: check each job completed by deadline
    job = scenario["jobs"]
    sla_violations = 0
    for _, row in job.iterrows():
        scheduled = result["job_schedule"][row["id"], row["release"]:row["deadline"] + 1].sum()
        if scheduled < row["energy_kwh"] - 1.0:
            sla_violations += 1

    green_share = float(result["green_use"].sum() /
                        (result["green_use"].sum() + result["grid"].sum() + 1e-9))

    return {
        "cost_$": cost,
        "carbon_kg": carbon_kg,
        "degradation_$": deg,
        "objective": total,
        "sla_violations": sla_violations,
        "green_share": green_share,
    }


# ---------------------------------------------------------------------------
# 1. MILP optimal
# ---------------------------------------------------------------------------
def solve_milp(scenario, carbon_weight=1.0, time_limit=20):
    T = scenario["T"]
    price = scenario["price"]
    carbon = scenario["carbon"]
    green = scenario["green"]
    infer = scenario["inference"]
    jobs = scenario["jobs"]
    sys = scenario["sys"]
    bat = scenario["battery"]

    n = len(jobs)
    model = pulp.LpProblem("co_sched", pulp.LpMinimize)

    # Variables
    x = [[pulp.LpVariable(f"x_{i}_{t}", lowBound=0) for t in range(T)] for i in range(n)]
    grid = [pulp.LpVariable(f"g_{t}", lowBound=0) for t in range(T)]
    chg = [pulp.LpVariable(f"chg_{t}", lowBound=0, upBound=bat["max_charge_kw"]) for t in range(T)]
    dis = [pulp.LpVariable(f"dis_{t}", lowBound=0, upBound=bat["max_discharge_kw"]) for t in range(T)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=bat["soc_min"], upBound=bat["capacity_kwh"]) for t in range(T)]
    green_use = [pulp.LpVariable(f"gu_{t}", lowBound=0) for t in range(T)]

    # Objective: energy cost + carbon cost + battery degradation
    obj = pulp.lpSum(
        price[t] * grid[t] * sys["dt_hours"]
        + carbon_weight * sys["carbon_price"] * carbon[t] * grid[t] * sys["dt_hours"] / 1000.0
        + bat["degradation_cost"] * (chg[t] + dis[t]) * sys["dt_hours"]
        for t in range(T)
    )
    model += obj

    # Constraints
    for t in range(T):
        job_load_t = pulp.lpSum(x[i][t] for i in range(n))
        # GPU capacity: inference + deferrable <= total
        model += job_load_t + infer[t] <= sys["gpu_total_kw"], f"gpu_{t}"
        # Power balance (EQUALITY): supply = demand
        # green_use + grid + discharge = inference + job_load + charge
        model += (green_use[t] + grid[t] + dis[t]
                  == infer[t] + job_load_t + chg[t]), f"bal_{t}"
        # Green use cannot exceed available
        model += green_use[t] <= green[t], f"grn_{t}"
        # Green use cannot exceed total demand (can't "use" more than we consume)
        model += green_use[t] <= infer[t] + job_load_t + chg[t], f"grnuse_{t}"
        # Grid limit
        model += grid[t] <= sys["grid_limit_kw"], f"gridlim_{t}"

    # Battery SOC dynamics
    for t in range(T):
        prev = bat["soc_init"] if t == 0 else soc[t - 1]
        model += soc[t] == prev + bat["eta_charge"] * chg[t] - dis[t] * (1.0 / bat["eta_discharge"]), f"soc_{t}"

    # Each job must complete its energy within [release, deadline]
    for _, row in jobs.iterrows():
        i = row["id"]
        model += pulp.lpSum(x[i][t] for t in range(row["release"], row["deadline"] + 1)) == row["energy_kwh"], f"job_{i}"
        for t in range(T):
            if t < row["release"] or t > row["deadline"]:
                model += x[i][t] == 0, f"window_{i}_{t}"

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    model.solve(solver)

    res = _empty_result(T, n)
    for i in range(n):
        for t in range(T):
            res["job_schedule"][i, t] = x[i][t].value() or 0.0
    for t in range(T):
        res["grid"][t] = grid[t].value() or 0.0
        res["charge"][t] = chg[t].value() or 0.0
        res["discharge"][t] = dis[t].value() or 0.0
        res["soc"][t] = soc[t].value() or 0.0
        res["green_use"][t] = green_use[t].value() or 0.0
        res["green_curtail"][t] = max(0.0, green[t] - res["green_use"][t])
        res["total_load"][t] = infer[t] + res["job_schedule"][:, t].sum()

    res["status"] = pulp.LpStatus[model.status]
    res["objective_milp"] = pulp.value(model.objective)
    return res


# ---------------------------------------------------------------------------
# 2. AFAP baseline: As-Fast-As-Possible (run jobs immediately at release)
# ---------------------------------------------------------------------------
def solve_afap(scenario):
    T = scenario["T"]
    jobs = scenario["jobs"]
    n = len(jobs)
    infer = scenario["inference"]
    gpu_cap = scenario["sys"]["gpu_total_kw"]
    green = scenario["green"]
    sys = scenario["sys"]
    bat = scenario["battery"]

    res = _empty_result(T, n)
    remaining = {row["id"]: row["energy_kwh"] for _, row in jobs.iterrows()}

    # Sort jobs by release time
    order = sorted(jobs.to_dict("records"), key=lambda r: r["release"])
    for job in order:
        i = job["id"]
        t = job["release"]
        while remaining[i] > 1e-6 and t <= job["deadline"]:
            available = gpu_cap - infer[t] - res["job_schedule"][:, t].sum()
            take = min(available, remaining[i])
            if take > 0:
                res["job_schedule"][i, t] = take
                remaining[i] -= take
            t += 1

    # Power balance: just buy from grid whatever we can't cover with green,
    # no battery smarts (charge only when green surplus, discharge never)
    soc = bat["soc_init"]
    for t in range(T):
        load = infer[t] + res["job_schedule"][:, t].sum()
        gu = min(green[t], load)
        res["green_use"][t] = gu
        net_load = load - gu
        res["grid"][t] = max(0, net_load)
        res["green_curtail"][t] = max(0, green[t] - gu)
        res["total_load"][t] = load
        soc = soc  # no battery use in AFAP
        res["soc"][t] = soc

    return res


# ---------------------------------------------------------------------------
# 3. TOU baseline: run deferrable jobs only in valley-price hours
# ---------------------------------------------------------------------------
def solve_tou(scenario):
    T = scenario["T"]
    jobs = scenario["jobs"]
    n = len(jobs)
    infer = scenario["inference"]
    gpu_cap = scenario["sys"]["gpu_total_kw"]
    price = scenario["price"]
    green = scenario["green"]

    res = _empty_result(T, n)
    # Valley hours: below 35th percentile price
    valley_thresh = np.percentile(price, 35)
    valley_hours = [t for t in range(T) if price[t] <= valley_thresh]

    remaining = {row["id"]: row["energy_kwh"] for _, row in jobs.iterrows()}
    # Sort by deadline (earliest deadline first)
    order = sorted(jobs.to_dict("records"), key=lambda r: r["deadline"])
    for job in order:
        i = job["id"]
        for t in range(T):
            if remaining[i] <= 1e-6:
                break
            if t not in valley_hours:
                continue
            if t < job["release"] or t > job["deadline"]:
                continue
            available = gpu_cap - infer[t] - res["job_schedule"][:, t].sum()
            take = min(available, remaining[i])
            if take > 0:
                res["job_schedule"][i, t] = take
                remaining[i] -= take

    # Fallback: if a job still has remaining (no valley hour fit), run ASAP
    for job in order:
        i = job["id"]
        t = job["release"]
        while remaining[i] > 1e-6 and t <= job["deadline"]:
            available = gpu_cap - infer[t] - res["job_schedule"][:, t].sum()
            take = min(available, remaining[i])
            if take > 0:
                res["job_schedule"][i, t] = take
                remaining[i] -= take
            t += 1

    for t in range(T):
        load = infer[t] + res["job_schedule"][:, t].sum()
        gu = min(green[t], load)
        res["green_use"][t] = gu
        res["grid"][t] = max(0, load - gu)
        res["green_curtail"][t] = max(0, green[t] - gu)
        res["total_load"][t] = load

    return res


# ---------------------------------------------------------------------------
# 4. Our greedy heuristic: "value-aware rolling schedule"
#    - Each hour, score available jobs by (urgency) * (avoided cost this hour)
#    - Battery: charge when price low / green surplus, discharge when price high
# ---------------------------------------------------------------------------
def solve_heuristic(scenario, carbon_weight=1.0):
    T = scenario["T"]
    jobs = scenario["jobs"]
    n = len(jobs)
    infer = scenario["inference"]
    gpu_cap = scenario["sys"]["gpu_total_kw"]
    price = scenario["price"]
    carbon = scenario["carbon"]
    green = scenario["green"]
    sys = scenario["sys"]
    bat = scenario["battery"]

    res = _empty_result(T, n)
    remaining = {row["id"]: row["energy_kwh"] for _, row in jobs.iterrows()}
    soc = bat["soc_init"]

    # Effective cost per hour = price + carbon_price * carbon * carbon_weight
    eff_cost = price + carbon_weight * sys["carbon_price"] * carbon / 1000.0
    # Rank hours by cheapness
    cheap_rank = np.argsort(eff_cost)  # hours from cheapest to most expensive

    # First pass: schedule jobs into cheapest feasible hours, EDF order
    order = sorted(jobs.to_dict("records"), key=lambda r: r["deadline"])
    for job in order:
        i = job["id"]
        # Candidate hours within window, sorted by eff_cost ascending
        window = [t for t in range(job["release"], job["deadline"] + 1)]
        window.sort(key=lambda t: eff_cost[t])
        for t in window:
            if remaining[i] <= 1e-6:
                break
            available = gpu_cap - infer[t] - res["job_schedule"][:, t].sum()
            if available > 0:
                # Distribute proportionally: fill cheap hours but don't overfill
                take = min(available, remaining[i])
                res["job_schedule"][i, t] = take
                remaining[i] -= take

    # Battery policy: after jobs placed, simulate
    for t in range(T):
        load = infer[t] + res["job_schedule"][:, t].sum()
        # Use green first
        gu = min(green[t], load)
        res["green_use"][t] = gu
        net = load - gu
        # Decide battery action based on price
        if eff_cost[t] <= np.percentile(eff_cost, 30) and soc < bat["capacity_kwh"] - 1:
            # Cheap: charge (from green surplus, or from grid if green already covers load)
            surplus_green = green[t] - gu
            chg = min(surplus_green, bat["max_charge_kw"], bat["capacity_kwh"] - soc)
            res["charge"][t] = chg
            soc += bat["eta_charge"] * chg
        elif eff_cost[t] >= np.percentile(eff_cost, 70) and soc > bat["soc_min"]:
            # Expensive: discharge to offset grid
            dis = min(bat["max_discharge_kw"], (soc - bat["soc_min"]) * bat["eta_discharge"], net)
            res["discharge"][t] = dis
            net -= dis
            soc -= dis / bat["eta_discharge"]
        res["grid"][t] = max(0, net)
        res["green_curtail"][t] = max(0, green[t] - gu - res["charge"][t])
        res["total_load"][t] = load
        res["soc"][t] = soc

    return res
