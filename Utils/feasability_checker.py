from collections import defaultdict
import numpy as np
from Utils.Generell.utils import boxed_print


def check_instance_feasibility_extended(R_p, Entry_p, Max_t, P, D, D_Full, T, W_coeff, verbose=True):
    """
    Enhanced feasibility check with detailed entry-day analysis.

    This function performs comprehensive feasibility checks BEFORE optimization:
    1. Entry day capacity vs. demand (CRITICAL - must be satisfied)
    2. Total capacity vs. total demand
    3. Individual patient schedulability
    4. Work cycle consistency

    Parameters:
    - R_p (dict): Patient requirements (number of treatments needed)
    - Entry_p (dict): Patient entry days
    - Max_t (dict): Therapist capacity as (therapist, day) -> capacity
    - P (list): List of patient IDs
    - D (list): Focus horizon days
    - D_Full (list): Full horizon days
    - T (list): List of therapist IDs
    - W_coeff (float): Work cycle coefficient
    - verbose (bool): If True, print detailed analysis

    Returns:
    - is_feasible (bool): True if instance is feasible
    - results (dict): Detailed analysis including problematic periods
    """

    if verbose:
        boxed_print("INSTANCE FEASIBILITY PRE-CHECK", width=100, center=True)

    results = {
        "is_feasible": True,
        "issues": [],
        "entry_day_analysis": {},
        "bottleneck_periods": [],
        "total_capacity_check": {},
        "patient_schedulability": {}
    }

    # Categorize patients
    P_Pre, P_F, P_Post, P_Join = categorize_patients(Entry_p, D)

    # ========================================================================
    # CRITICAL CHECK 1: Entry Day Capacity vs. Demand
    # ========================================================================
    if verbose:
        boxed_print("CHECK 1: Entry Day Capacity vs. Demand", width=100, center=False)

    entry_day_demand = defaultdict(int)
    entry_day_patients = defaultdict(list)

    for p in P:
        entry_day = Entry_p[p]
        if entry_day in D_Full:
            entry_day_demand[entry_day] += 1  # Each patient needs exactly 1 slot on entry
            entry_day_patients[entry_day].append(p)

    for d in sorted(entry_day_demand.keys()):
        total_capacity = sum(Max_t.get((t, d), 0) for t in T)
        demand = entry_day_demand[d]
        deficit = max(0, demand - total_capacity)

        results["entry_day_analysis"][d] = {
            "demand": demand,
            "capacity": total_capacity,
            "deficit": deficit,
            "patients": entry_day_patients[d],
            "feasible": demand <= total_capacity
        }

        if demand > total_capacity:
            results["is_feasible"] = False
            issue = (f"Period {d}: {demand} patients must start, "
                     f"but only {total_capacity} slots available (DEFICIT: {deficit})")
            results["issues"].append(issue)
            results["bottleneck_periods"].append(d)

            if verbose:
                boxed_print(f"❌ {issue}", width=100, center=False)
                affected_str = f"   Affected patients: {entry_day_patients[d][:15]}"
                if len(entry_day_patients[d]) > 15:
                    affected_str += f"... (+{len(entry_day_patients[d]) - 15} more)"
                boxed_print(affected_str, width=100, center=False)
        else:
            slack = total_capacity - demand
            if verbose:
                status = "✅" if slack >= 2 else "⚠️ " if slack >= 1 else "🔥"
                boxed_print(f"{status} Period {d}: {demand} patients, {total_capacity} slots (slack: {slack})",
                            width=100, center=False)

    # ========================================================================
    # CHECK 2: Total Capacity vs. Total Demand
    # ========================================================================
    if verbose:
        boxed_print("\nCHECK 2: Total Capacity vs. Total Demand", width=100, center=False)

    total_capacity = sum(Max_t.get((t, d), 0) for t in T for d in D_Full)
    # Pre and Focus patients need full requirements, Post patients need only entry treatment
    total_demand = sum(R_p[p] for p in P_Pre + P_F) + len(P_Post)

    results["total_capacity_check"] = {
        "total_capacity": total_capacity,
        "total_demand": total_demand,
        "utilization": total_demand / total_capacity if total_capacity > 0 else float('inf'),
        "feasible": total_demand <= total_capacity
    }

    if total_demand > total_capacity:
        results["is_feasible"] = False
        issue = f"Total demand ({total_demand}) exceeds total capacity ({total_capacity})"
        results["issues"].append(issue)
        if verbose:
            boxed_print(f"❌ {issue}", width=100, center=False)
    else:
        utilization_pct = (total_demand / total_capacity * 100) if total_capacity > 0 else 0
        if verbose:
            boxed_print(f"✅ Total capacity: {total_capacity}, Total demand: {total_demand} "
                        f"(Utilization: {utilization_pct:.1f}%)", width=100, center=False)

    # ========================================================================
    # CHECK 3: Individual Patient Schedulability (Focus patients only)
    # ========================================================================
    if verbose:
        boxed_print("\nCHECK 3: Patient Schedulability", width=100, center=False)

    unschedulable_patients = []
    for p in P_F:  # Only check Focus patients (Pre are already treated, Post don't need discharge)
        entry_day = Entry_p[p]
        req = R_p[p]

        # Days available for this patient
        available_days = [d for d in D_Full if d >= entry_day]
        # Capacity available after entry
        available_capacity = sum(Max_t.get((t, d), 0) for t in T for d in available_days)

        # Rough estimate: can patient be scheduled?
        if len(available_days) < req or available_capacity < req:
            results["is_feasible"] = False
            issue = f"Patient {p}: Needs {req} treatments, only {len(available_days)} days and {available_capacity} capacity available"
            results["issues"].append(issue)
            results["patient_schedulability"][p] = {
                "requirements": req,
                "available_days": len(available_days),
                "available_capacity": available_capacity,
                "feasible": False
            }
            unschedulable_patients.append(p)

            if verbose:
                boxed_print(f"❌ {issue}", width=100, center=False)

    if unschedulable_patients and verbose:
        boxed_print(f"⚠️  {len(unschedulable_patients)} Focus patients cannot be scheduled",
                    width=100, center=False)
    elif verbose:
        boxed_print(f"✅ All {len(P_F)} Focus patients are potentially schedulable",
                    width=100, center=False)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    if verbose:
        boxed_print("\n" + "=" * 100, width=100, center=False, border="")
        if results["is_feasible"]:
            boxed_print("✅ INSTANCE IS FEASIBLE - All checks passed!", width=100, center=True)
        else:
            boxed_print("❌ INSTANCE IS INFEASIBLE", width=100, center=True)
            boxed_print(f"\nFound {len(results['issues'])} critical issues:", width=100, center=False)
            for idx, issue in enumerate(results["issues"], 1):
                boxed_print(f"  {idx}. {issue}", width=100, center=False)
        boxed_print("=" * 100, width=100, center=False, border="")

    return results["is_feasible"], results


def repair_infeasible_instance(R_p, Entry_p, Max_t, feas_results, D_Full, T):
    """
    Attempts to repair an infeasible instance by adjusting capacity or entry days.

    Strategy:
    1. Identify bottleneck periods from feas_results
    2. Try to shift patients to adjacent periods if possible
    3. If shifting not possible, increase capacity at bottleneck periods

    Parameters:
    - R_p (dict): Patient requirements
    - Entry_p (dict): Patient entry days
    - Max_t (dict): Therapist capacity
    - feas_results (dict): Results from feasibility check
    - D_Full (list): Full planning horizon
    - T (list): List of therapists

    Returns:
    - R_p (dict): Unchanged requirements
    - Entry_p_repaired (dict): Potentially modified entry days
    - Max_t_repaired (dict): Potentially modified capacity
    """

    boxed_print("🔧 ATTEMPTING AUTOMATIC INSTANCE REPAIR", width=100, center=True)

    bottleneck_periods = feas_results.get("bottleneck_periods", [])

    if not bottleneck_periods:
        boxed_print("No bottlenecks detected - nothing to repair", width=100)
        return R_p, Entry_p, Max_t

    Max_t_repaired = Max_t.copy()
    Entry_p_repaired = Entry_p.copy()

    total_shifts = 0
    total_capacity_added = 0

    for d in bottleneck_periods:
        analysis = feas_results["entry_day_analysis"][d]
        deficit = analysis["deficit"]
        patients_at_d = analysis["patients"]

        boxed_print(f"\n📍 Repairing period {d} (deficit: {deficit}, patients: {len(patients_at_d)})",
                    width=100, center=False)

        # ====================================================================
        # STRATEGY 1: Try to shift patients to adjacent periods
        # ====================================================================
        shifted = 0
        shift_attempts = [(d - 1, "backward"), (d + 1, "forward")]

        for p in patients_at_d[:deficit * 2]:  # Try shifting up to 2x deficit
            if shifted >= deficit:
                break

            for shift_d, direction in shift_attempts:
                if shift_d not in D_Full or shift_d < 1:
                    continue

                # Calculate current demand at shift_d
                shift_capacity = sum(Max_t_repaired.get((t, shift_d), 0) for t in T)
                shift_demand = sum(1 for pp in R_p if Entry_p_repaired.get(pp) == shift_d)

                if shift_demand < shift_capacity:
                    print(f"  ↔️  Shifting patient {p}: period {d} → {shift_d} ({direction})")
                    Entry_p_repaired[p] = shift_d
                    shifted += 1
                    total_shifts += 1
                    break

        if shifted > 0:
            print(f"  ✅ Successfully shifted {shifted} patients away from period {d}")

        # ====================================================================
        # STRATEGY 2: If shifting didn't solve it, increase capacity
        # ====================================================================
        remaining_deficit = deficit - shifted

        if remaining_deficit > 0:
            print(f"  ⚠️  Still {remaining_deficit} patients cannot be shifted")
            print(f"  💡 Increasing capacity for period {d} by {remaining_deficit}")

            # Distribute extra capacity across therapists (round-robin)
            per_therapist = remaining_deficit // len(T)
            remainder = remaining_deficit % len(T)

            for idx, t in enumerate(T):
                extra = per_therapist + (1 if idx < remainder else 0)
                if extra > 0:
                    old_cap = Max_t_repaired.get((t, d), 0)
                    Max_t_repaired[(t, d)] = old_cap + extra
                    print(f"     Therapist {t} at period {d}: {old_cap} → {old_cap + extra} (+{extra})")
                    total_capacity_added += extra

    # Summary
    boxed_print("\n" + "=" * 100, width=100, center=False, border="")
    boxed_print("REPAIR SUMMARY", width=100, center=True)
    boxed_print(f"Patients shifted: {total_shifts}", width=100, center=False)
    boxed_print(f"Capacity added: {total_capacity_added} slots", width=100, center=False)
    boxed_print("=" * 100, width=100, center=False, border="")

    return R_p, Entry_p_repaired, Max_t_repaired


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


# Keep the original function for backward compatibility
def check_instance_feasibility(R_p, Entry_p, Max_t, P, D, D_Full, T, W_coeff, app_data, verbose=True):
    """
    Original function - redirects to extended version.
    Kept for backward compatibility.
    """
    return check_instance_feasibility_extended(R_p, Entry_p, Max_t, P, D, D_Full, T, W_coeff, verbose)