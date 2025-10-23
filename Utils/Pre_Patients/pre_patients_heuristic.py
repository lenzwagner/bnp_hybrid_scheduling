import math
import copy
from collections import defaultdict


def compute_app_effectiveness(app_count, theta_type, params):
    """Compute AI effectiveness based on learning curve type."""
    if theta_type == 'lin':
        theta_base = params['theta_base']
        lin_increase = params['lin_increase']
        eff = theta_base + lin_increase * app_count
        return min(eff, 1)
    elif theta_type == 'exp':
        theta_base = params['theta_base']
        k = params['k_learn']
        eff = theta_base + (1 - theta_base) * (1 - math.exp(-k * app_count))
        return min(eff, 1)
    elif theta_type == 'sigmoid':
        theta_base = params['theta_base']
        k = params['k_learn']
        infl_point = params['infl_point']
        eff = theta_base + (1 - theta_base) / (1 + math.exp(-k * (app_count - infl_point)))
        return min(eff, 1)
    else:
        return params['const']


def pre_processing_schedule(P, P_F, T, D, Entry_p, CReq_p, theta_app_type,
                            learning_params, TS, FS, Max_t_input, nr_c, T_mapping):
    """
    Enhanced pre-patient scheduling with smart capacity reservation and conflict resolution.

    Key improvements:
    1. Smart therapist selection (load balancing)
    2. Look-ahead for bottleneck detection
    3. Conflict resolution via patient shifting
    4. Fallback mechanisms
    """

    # ============================================================================
    # FEASIBILITY PRE-CHECK (from previous code)
    # ============================================================================
    from Utils.Generell.utils import boxed_print

    boxed_print("PRE-PATIENT SCHEDULING: FEASIBILITY PRE-CHECK", width=100, center=True)

    entry_demand = defaultdict(int)
    entry_patients = defaultdict(list)

    for p in P:
        entry_day = Entry_p[p]
        entry_demand[entry_day] += nr_c[p]
        entry_patients[entry_day].append(p)

    feasible = True
    bottlenecks = []

    for d in sorted(entry_demand.keys()):
        demand = entry_demand[d]
        capacity = sum(Max_t_input.get((t, d), 0) for t in T)

        if demand > capacity:
            deficit = demand - capacity
            boxed_print(
                f"❌ INFEASIBLE: Period {d} requires {demand} slots, only {capacity} available (deficit: {deficit})",
                width=100, center=False
            )
            boxed_print(f"   Affected patients: {entry_patients[d]}", width=100, center=False)
            feasible = False
            bottlenecks.append(d)
        else:
            slack = capacity - demand
            status = "✅" if slack >= 2 else "⚠️ "
            boxed_print(f"{status} Period {d}: {demand} demand, {capacity} capacity (slack: {slack})",
                        width=100, center=False)

    if not feasible:
        boxed_print("\n" + "=" * 100, width=100, center=False, border="")
        boxed_print("❌ PRE-PATIENT SCHEDULING IS STRUCTURALLY INFEASIBLE!", width=100, center=True)
        boxed_print("=" * 100, width=100, center=False, border="")
        return 'Infeasible'

    boxed_print("✅ PRE-CHECK PASSED - PROCEEDING WITH SCHEDULING", width=100, center=True)
    boxed_print("=" * 100 + "\n", width=100, center=False, border="")

    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    Max_t = copy.deepcopy(Max_t_input)
    x, y, LOS = {}, {}, {}
    cumEff_dict = {}
    therapist_assignments = {}
    therapist_load = defaultdict(int)  # Track total load per therapist
    app_count = {p: 0 for p in P}
    LOS_total = 0

    # ============================================================================
    # STEP 0: SMART CAPACITY RESERVATION FOR FOCUS PATIENTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 0: Reserving capacity for Focus patients (P_F)")
    print("=" * 80)

    # Sort P_F by entry day to handle earlier patients first
    P_F_sorted = sorted(P_F, key=lambda p: Entry_p[p])

    reservation_failed = []

    for pf in P_F_sorted:
        entry_day = Entry_p[pf]
        required_slots = nr_c[pf]

        # Find therapist with sufficient capacity AND minimal load
        candidates = []
        for t in T:
            available = Max_t.get((t, entry_day), 0)
            if available >= required_slots:
                # Score: prefer therapists with more capacity AND lower total load
                score = available * 1000 - therapist_load[t]  # Prioritize capacity, then load
                candidates.append((score, t, available))

        if candidates:
            # Select best therapist
            candidates.sort(reverse=True)  # Highest score first
            _, t_best, available = candidates[0]

            # Reserve capacity
            Max_t[(t_best, entry_day)] -= required_slots
            therapist_load[t_best] += required_slots

            print(f"✅ Reserved: Patient {pf} → Therapist {t_best}, Day {entry_day}, "
                  f"Slots: {required_slots} (Remaining: {Max_t[(t_best, entry_day)]})")
        else:
            print(f"⚠️  Cannot reserve for patient {pf} at day {entry_day} (need {required_slots} slots)")
            reservation_failed.append(pf)

    if reservation_failed:
        print(f"\n⚠️  WARNING: {len(reservation_failed)} Focus patients could not be reserved:")
        print(f"   {reservation_failed}")
        print("   Attempting conflict resolution...\n")

        # Try to shift some reserved patients to make room
        for pf in reservation_failed:
            entry_day = Entry_p[pf]
            required_slots = nr_c[pf]

            # Try adjacent days
            for shift_day in [entry_day - 1, entry_day + 1]:
                if shift_day < min(D) or shift_day > max(D):
                    continue

                # Check if shifting here would work
                available = sum(Max_t.get((t, shift_day), 0) for t in T)
                if available >= required_slots:
                    print(f"   ↔️  Shifting patient {pf}: {entry_day} → {shift_day}")
                    Entry_p[pf] = shift_day

                    # Now reserve at new day
                    candidates = [(Max_t.get((t, shift_day), 0), t) for t in T
                                  if Max_t.get((t, shift_day), 0) >= required_slots]
                    if candidates:
                        candidates.sort(reverse=True)
                        _, t_best = candidates[0]
                        Max_t[(t_best, shift_day)] -= required_slots
                        therapist_load[t_best] += required_slots
                        print(f"   ✅ Shifted and reserved at day {shift_day}")
                        break
            else:
                # Could not shift
                print(f"   ❌ CRITICAL: Cannot schedule patient {pf} even after shifting attempts!")
                return 'Infeasible'

    print("\n✅ All Focus patients successfully reserved\n")

    # ============================================================================
    # STEP 1: SCHEDULE EACH PATIENT
    # ============================================================================
    print("=" * 80)
    print("STEP 1: Scheduling all patients")
    print("=" * 80 + "\n")

    for p in P:
        print(f"Scheduling patient {p} (Entry: {Entry_p[p]}, Req: {CReq_p[p]}, Count: {nr_c[p]})")

        d_current = Entry_p[p]
        cum_effective = 0

        # === First session (mandatory at entry day) ===
        candidates = []
        for t in T:
            available = Max_t.get((t, d_current), 0)
            if available >= nr_c[p]:
                # Prefer therapist with more capacity (load balancing)
                score = available
                candidates.append((score, t, available))

        if not candidates:
            print(f"❌ ERROR: No therapist available for patient {p} at entry day {d_current}")
            print(f"   Required: {nr_c[p]}, Available capacity:")
            for t in T:
                print(f"      Therapist {t}: {Max_t.get((t, d_current), 0)}")
            return 'Infeasible'

        # Select best therapist
        candidates.sort(reverse=True)
        _, assigned_therapist, _ = candidates[0]

        # Assign first session
        Max_t[(assigned_therapist, d_current)] -= nr_c[p]
        therapist_load[assigned_therapist] += nr_c[p]
        x[(p, assigned_therapist, d_current)] = nr_c[p]
        cum_effective += 1
        therapist_assignments[p] = assigned_therapist

        print(f"  Day {d_current}: Therapy with T{assigned_therapist} (first session). "
              f"CumEff: {cum_effective:.2f}")

        d_current += 1

        # === Subsequent sessions ===
        while cum_effective < CReq_p[p]:
            if d_current > max(D):
                print(f"  ⚠️  Patient {p} could not be fully treated within horizon")
                break

            t = assigned_therapist
            theta_app_eff = compute_app_effectiveness(app_count[p], theta_app_type, learning_params)

            # Check FS constraint (minimum sessions per window)
            window_start = max(Entry_p[p], d_current - TS + 1)
            window_end = d_current
            full_window = (window_end - window_start + 1) >= TS

            if full_window:
                therapy_count = sum(1 for d in range(window_start, window_end + 1)
                                    if any((p, tt, d) in x for tt in T))

                if therapy_count < FS and Max_t.get((t, d_current), 0) >= nr_c[p]:
                    # Must give therapy (FS rule)
                    x[(p, t, d_current)] = nr_c[p]
                    Max_t[(t, d_current)] -= nr_c[p]
                    therapist_load[t] += nr_c[p]
                    cum_effective += 1
                    print(f"  Day {d_current}: Therapy (FS rule). CumEff: {cum_effective:.2f}")
                else:
                    # Can give AI
                    y[(p, d_current)] = 1
                    app_count[p] += 1
                    cum_effective += theta_app_eff
                    print(f"  Day {d_current}: AI. CumEff: {cum_effective:.2f}, AppEff: {theta_app_eff:.2f}")
            else:
                # Accelerate discharge if close to completion
                if (CReq_p[p] - cum_effective) >= 1 and Max_t.get((t, d_current), 0) >= nr_c[p]:
                    x[(p, t, d_current)] = nr_c[p]
                    Max_t[(t, d_current)] -= nr_c[p]
                    therapist_load[t] += nr_c[p]
                    cum_effective += 1
                    print(f"  Day {d_current}: Therapy (accelerating). CumEff: {cum_effective:.2f}")
                else:
                    y[(p, d_current)] = 1
                    app_count[p] += 1
                    cum_effective += theta_app_eff
                    print(f"  Day {d_current}: AI. CumEff: {cum_effective:.2f}, AppEff: {theta_app_eff:.2f}")

            d_current += 1

        # === Ensure last treatment is therapy ===
        if cum_effective >= CReq_p[p]:
            last_days = [d for (pp, tt, d) in x if pp == p] + [d for (pp, d) in y if pp == p]
            last_day = max(last_days) if last_days else Entry_p[p]

            if not any((p, tt, last_day) in x for tt in T):
                print(f"  ⚠️  Last session is AI, attempting to switch...")

                if Max_t.get((t, last_day), 0) >= nr_c[p]:
                    # Switch last AI to therapy
                    x[(p, t, last_day)] = nr_c[p]
                    Max_t[(t, last_day)] -= nr_c[p]
                    therapist_load[t] += nr_c[p]
                    if (p, last_day) in y:
                        del y[(p, last_day)]
                        app_count[p] -= 1
                    cum_effective = cum_effective - theta_app_eff + 1
                    print(f"  ✅ Switched to therapy at day {last_day}")
                else:
                    # Extend to next day
                    print(f"  ⚠️  Cannot switch, extending to next day...")
                    d_extend = last_day + 1
                    while d_extend <= max(D):
                        if Max_t.get((t, d_extend), 0) >= nr_c[p]:
                            x[(p, t, d_extend)] = nr_c[p]
                            Max_t[(t, d_extend)] -= nr_c[p]
                            therapist_load[t] += nr_c[p]
                            print(f"  ✅ Extended therapy to day {d_extend}")
                            last_day = d_extend
                            break
                        else:
                            # Give AI while searching
                            y[(p, d_extend)] = 1
                            app_count[p] += 1
                            theta_temp = compute_app_effectiveness(app_count[p], theta_app_type, learning_params)
                            cum_effective += theta_temp
                            print(f"  Day {d_extend}: AI (searching for therapy slot). CumEff: {cum_effective:.2f}")
                        d_extend += 1

                    if d_extend > max(D):
                        print(f"  ❌ Could not assign final therapy within horizon for patient {p}")

            LOS[p] = last_day - Entry_p[p] + 1
            cumEff_dict[p] = cum_effective
            LOS_total += LOS[p]
            print(f"  ✓ Patient {p} completed: LOS={LOS[p]}, CumEff={cum_effective:.2f}\n")

    print(f"\n{'=' * 80}")
    print(f"SCHEDULING COMPLETE: Total LOS = {LOS_total}")
    print(f"{'=' * 80}\n")

    # Filter and transform
    filtered_x = {(p, t, d): v for (p, t, d), v in x.items() if d > 0}
    transformed_x = transform_t_values(filtered_x, T_mapping)

    return x, y, LOS, transformed_x, filtered_x


def transform_t_values(original_dict, t_mapping):
    """Transform therapist indices according to mapping."""
    from collections import defaultdict
    transformed_dict = defaultdict(int)

    for (p, t, d), value in original_dict.items():
        new_t = t_mapping.get(t, t)
        new_key = (p, new_t, d)
        transformed_dict[new_key] += value

    return dict(transformed_dict)
