from Utils.Generell.utils import *

def check_instance_feasibility(R_p, Entry_p, Max_t, P, D, D_Full, T, W_coeff, app_data, verbose=True):
    """
    Check if a generated instance is feasible for the patient scheduling problem.
    Accounts for Post-patients not requiring discharge within the horizon.

    Parameters:
    - R_p (dict): Patient requirements (number of treatments needed)
    - Entry_p (dict): Patient entry days
    - Max_t (dict): Therapist capacity as (therapist, day) -> capacity (treatment slots)
    - P (list): List of patient IDs
    - D (list): Focus horizon days
    - D_Ext (list): Extended horizon days
    - D_Full (list): Full horizon days
    - T (list): List of therapist IDs
    - M_p (dict): Mean LOS per patient
    - W_coeff (float): Work cycle coefficient (W_on / (W_on + W_off))
    - app_data (dict): Application data with parameters like W, W_min
    - verbose (bool): If True, boxed_print detailed feasibility analysis

    Returns:
    - bool: True if the instance is feasible, False otherwise
    - dict: Dictionary with feasibility check results and details
    """
    # Categorize patients into Pre, Focus, and Post
    P_Pre, P_F, P_Post, P_Join = categorize_patients(Entry_p, D)

    results = {
        "is_feasible": True,
        "issues": [],
        "entry_day_check": {},
        "capacity_check": {},
        "los_check": {},
        "total_demand_vs_capacity": {},
        "work_cycle_check": {}
    }

    # 1. Check therapist capacity on entry days (applies to all patients)
    if verbose:
        boxed_print("\n=== Checking Therapist Capacity on Entry Days ===")
    entry_day_counts = defaultdict(int)
    for p in P:
        entry_day = Entry_p[p]
        entry_day_counts[entry_day] += 1  # Each patient needs one treatment slot on entry day

    for d in sorted(set(Entry_p.values()) & set(D_Full)):  # Only check days with patient entries
        total_capacity = sum(Max_t.get((t, d), 0) for t in T)
        patients_entering = entry_day_counts.get(d, 0)
        if patients_entering > total_capacity:
            results["is_feasible"] = False
            issue = (f"Entry day {d}: {patients_entering} patients enter, but only "
                     f"{total_capacity} treatment slots available.")
            results["issues"].append(issue)
            results["entry_day_check"][d] = {
                "patients": patients_entering,
                "capacity": total_capacity,
                "feasible": False
            }
            if verbose:
                boxed_print(issue)
        else:
            results["entry_day_check"][d] = {
                "patients": patients_entering,
                "capacity": total_capacity,
                "feasible": True
            }
            if verbose:
                boxed_print(f"Entry day {d}: {patients_entering} patients, {total_capacity} slots - Feasible")

    # 2. Check total therapist capacity vs. total patient demand
    # For Pre and Focus patients: full requirements (R_p[p])
    # For Post patients: only entry day treatment (1 treatment)
    if verbose:
        boxed_print("\n=== Checking Total Capacity vs. Demand ===")
    total_capacity = sum(Max_t.get((t, d), 0) for t in T for d in D_Full)
    total_demand = sum(R_p[p] for p in P_Pre + P_F) + len(P_Post)  # Post-patients need 1 treatment
    results["total_demand_vs_capacity"] = {
        "total_capacity": total_capacity,
        "total_demand": total_demand,
        "feasible": total_demand <= total_capacity
    }
    if total_demand > total_capacity:
        results["is_feasible"] = False
        issue = (f"Total demand ({total_demand} treatments) exceeds total capacity "
                 f"({total_capacity} treatments).")
        results["issues"].append(issue)
        if verbose:
            boxed_print(issue)
    else:
        if verbose:
            boxed_print(f"Total capacity: {total_capacity}, Total demand: {total_demand} - Feasible")

    # 3. Check LOS constraints for Pre and Focus patients only
    if verbose:
        boxed_print("\n=== Checking LOS Constraints (Pre and Focus Patients) ===")
    W = app_data["MS"][0]
    W_min = app_data["MS_min"][0]
    for p in P_Pre + P_F:  # Exclude Post-patients
        entry_day = Entry_p[p]
        req = R_p[p]
        # Calculate feasible days for scheduling
        available_days = [d for d in D_Full if d >= entry_day]
        available_capacity = sum(Max_t.get((t, d), 0) for t in T for d in available_days)
        # Estimate minimum days needed based on W_min requirement
        min_days_needed = int(np.ceil(req / W_min)) if W_min > 0 else len(D_Full)
        if len(available_days) < min_days_needed:
            results["is_feasible"] = False
            issue = (f"Patient {p}: Requires {req} treatments, but only {len(available_days)} "
                     f"days available after entry day {entry_day} (min {min_days_needed} needed).")
            results["issues"].append(issue)
            results["los_check"][p] = {
                "requirements": req,
                "available_days": len(available_days),
                "min_days_needed": min_days_needed,
                "feasible": False
            }
            if verbose:
                boxed_print(issue)
        elif available_capacity < req:
            results["is_feasible"] = False
            issue = (f"Patient {p}: Requires {req} treatments, but only {available_capacity} "
                     f"treatment slots available after entry day {entry_day}.")
            results["issues"].append(issue)
            results["los_check"][p] = {
                "requirements": req,
                "available_days": len(available_days),
                "min_days_needed": min_days_needed,
                "available_capacity": available_capacity,
                "feasible": False
            }
            if verbose:
                boxed_print(issue)
        else:
            results["los_check"][p] = {
                "requirements": req,
                "available_days": len(available_days),
                "min_days_needed": min_days_needed,
                "available_capacity": available_capacity,
                "feasible": True
            }
            if verbose:
                boxed_print(f"Patient {p}: {req} treatments, {len(available_days)} days, "
                      f"{available_capacity} slots - Feasible")

    # Note for Post-patients
    if P_Post and verbose:
        boxed_print(f"\n=== Note on Post-Patients ===")
        boxed_print(f"Post-patients {P_Post} do not require discharge within the horizon. "
              f"Only their entry day treatment is considered in demand.")

    # 4. Check work cycle constraints
    if verbose:
        boxed_print("\n=== Checking Work Cycle Constraints ===")
    working_days = sum(1 for d in D_Full for t in T if Max_t.get((t, d), 0) > 0)
    expected_working_days = len(D_Full) * len(T) * W_coeff
    work_cycle_ratio = working_days / (len(D_Full) * len(T)) if len(D_Full) > 0 and len(T) > 0 else 0
    results["work_cycle_check"] = {
        "working_days": working_days,
        "expected_working_days": expected_working_days,
        "work_cycle_ratio": work_cycle_ratio,
        "feasible": abs(work_cycle_ratio - W_coeff) < 0.1  # Allow 10% deviation
    }
    if abs(work_cycle_ratio - W_coeff) >= 0.1:
        results["is_feasible"] = False
        issue = (f"Work cycle ratio {work_cycle_ratio:.2f} deviates significantly from "
                 f"expected {W_coeff:.2f}.")
        results["issues"].append(issue)
        if verbose:
            boxed_print(issue)
    else:
        if verbose:
            boxed_print(f"Work cycle ratio: {work_cycle_ratio:.2f}, Expected: {W_coeff:.2f} - Feasible")

    # Final summary
    if verbose:
        boxed_print("\n=== Feasibility Summary ===")
        boxed_print(f"Instance is {'feasible' if results['is_feasible'] else 'infeasible'}")
        boxed_print(f"Pre-patients: {P_Pre}")
        boxed_print(f"Focus-patients: {P_F}")
        boxed_print(f"Post-patients: {P_Post}")
        if not results["is_feasible"]:
            boxed_print("Issues found:")
            for issue in results["issues"]:
                boxed_print(f"- {issue}")
        else:
            boxed_print("No feasibility issues found.")

    return results["is_feasible"], results

# Required categorize_patients function
def categorize_patients(Entry_p, D):
    """
    Categorize patients into Pre, Focus, and Post groups based on entry days.

    Parameters:
    - Entry_p (dict): Patient entry days
    - D (list): Focus horizon days

    Returns:
    - tuple: (P_Pre, P_F, P_Post, P_Join)
    """
    P_Pre = [p for p, d in Entry_p.items() if d < min(D)]
    P_F = [p for p, d in Entry_p.items() if min(D) <= d <= max(D)]
    P_Post = [p for p, d in Entry_p.items() if d > max(D)]
    P_Join = sorted(P_F + P_Post)
    return P_Pre, P_F, P_Post, P_Join