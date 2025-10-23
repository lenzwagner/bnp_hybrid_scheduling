def initial_cg_starting_sol(max_capacity, patients, days, therapists, required_resources, entry_days, pre_assignments,
                            capacity_multipliers, flexible_patients, M_p, therapist_to_type=None):
    """
    Enhanced initial column generation with smart scheduling and conflict resolution.

    Key improvements:
    1. Feasibility pre-check before scheduling
    2. Smart therapist selection (load balancing)
    3. Multi-pass scheduling with backtracking
    4. Comprehensive conflict resolution

    Returns:
    - result_dict: dictionary with keys (p, t, d, 1) and assigned values
    - length_of_stay: dictionary with keys (p)
    - y, z, App, S, l: auxiliary dictionaries
    - remaining_capacity_td: remaining capacity per (t, d)
    - remaining_capacity_d: remaining capacity per day
    """

    # ============================================================================
    # FEASIBILITY PRE-CHECK (from previous code)
    # ============================================================================
    from collections import defaultdict

    print("\n" + "=" * 100)
    print("INITIAL COLUMN GENERATION: FEASIBILITY PRE-CHECK".center(100))
    print("=" * 100 + "\n")

    entry_demand = defaultdict(int)
    entry_patients_dict = defaultdict(list)

    for p in patients:
        entry_day = entry_days[p]
        entry_demand[entry_day] += capacity_multipliers[p]
        entry_patients_dict[entry_day].append(p)

    # Calculate available capacity AFTER pre-assignments
    capacity_copy = max_capacity.copy()

    print("Accounting for pre-assignments...")
    for (p, t, d), value in pre_assignments.items():
        if d > 0:
            if therapist_to_type and t in therapist_to_type:
                j = therapist_to_type[t]
                old_cap = capacity_copy.get((j, d), 0)
                capacity_copy[(j, d)] = old_cap - value
                if value > 0:
                    print(f"  Pre-assigned: Patient {p}, T{t}→Type{j}, Day {d}, Value {value}")
            else:
                old_cap = capacity_copy.get((t, d), 0)
                capacity_copy[(t, d)] = old_cap - value
                if value > 0:
                    print(f"  Pre-assigned: Patient {p}, T{t}, Day {d}, Value {value}")

    print("\nChecking entry day feasibility...\n")

    infeasible_periods = []
    for d in sorted(entry_demand.keys()):
        if d <= 0:
            continue

        demand = entry_demand[d]
        available = sum(capacity_copy.get((t, d), 0) for t in therapists)

        if demand > available:
            deficit = demand - available
            print(f"❌ Period {d}: Demand={demand}, Available={available} (DEFICIT: {deficit})")
            print(f"   Patients: {entry_patients_dict[d][:10]}"
                  f"{'...' if len(entry_patients_dict[d]) > 10 else ''}")
            infeasible_periods.append(d)
        else:
            slack = available - demand
            status = "✅" if slack >= 2 else "⚠️ " if slack >= 1 else "🔥"
            print(f"{status} Period {d}: Demand={demand}, Available={available} (slack: {slack})")

    if infeasible_periods:
        print("\n" + "=" * 100)
        print("❌ CRITICAL: INITIAL COLUMN GENERATION IS INFEASIBLE!".center(100))
        print("=" * 100)
        raise ValueError(
            f"Initial column generation impossible: insufficient capacity at periods {infeasible_periods}\n"
            f"Please adjust instance parameters before running optimization."
        )

    print("\n" + "=" * 100)
    print("✅ FEASIBILITY CHECK PASSED - GENERATING INITIAL COLUMNS".center(100))
    print("=" * 100 + "\n")

    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    result_dict = {}
    capacity = {(t, d): v for (t, d), v in capacity_copy.items() if d > 0}
    pre_assignments_filtered = {(p, t, d): v for (p, t, d), v in pre_assignments.items() if d > 0}

    # Apply pre-assignments to capacity
    if therapist_to_type is not None:
        for (p, t, d), value in pre_assignments_filtered.items():
            if t in therapist_to_type:
                j = therapist_to_type[t]
                if (j, d) in capacity:
                    capacity[(j, d)] -= value
    else:
        for (p, t, d), value in pre_assignments_filtered.items():
            if (t, d) in capacity:
                capacity[(t, d)] -= value

    # Initialize result_dict
    for p in patients:
        for t in therapists:
            for d in days:
                result_dict[(p, t, d, 1)] = 0

    # Track therapist load for load balancing
    therapist_load = defaultdict(int)
    scheduled_patients = []
    unscheduled_patients = []

    # ============================================================================
    # PASS 1: SMART INITIAL SCHEDULING
    # ============================================================================
    print("=" * 100)
    print("PASS 1: Smart Initial Scheduling".center(100))
    print("=" * 100 + "\n")

    # Sort patients: higher priority first (larger multiplier, earlier entry)
    sorted_patients = sorted(patients,
                             key=lambda p: (-capacity_multipliers[p], entry_days[p]))

    for p in sorted_patients:
        entry_day = entry_days[p]
        required_res = required_resources[p]
        capacity_multiplier = capacity_multipliers[p]

        print(f"Patient {p}: Entry={entry_day}, Req={required_res}, Mult={capacity_multiplier}")

        # === Find best therapist for entry day ===
        candidates = []
        for t in therapists:
            available = capacity.get((t, entry_day), 0)
            if available >= capacity_multiplier:
                # Score: balance between capacity and current load
                score = available * 100 - therapist_load[t]
                candidates.append((score, t, available))

        if not candidates:
            print(f"  ⚠️  Cannot schedule on entry day {entry_day}")
            unscheduled_patients.append(p)
            continue

        # Select best therapist
        candidates.sort(reverse=True)
        _, assigned_therapist, _ = candidates[0]

        # Assign first session
        result_dict[(p, assigned_therapist, entry_day, 1)] = 1
        capacity[(assigned_therapist, entry_day)] -= capacity_multiplier
        therapist_load[assigned_therapist] += capacity_multiplier
        assigned_resources = 1

        print(f"  ✓ Entry: T{assigned_therapist}, Day {entry_day}")

        # === Assign remaining sessions ===
        current_day_idx = days.index(entry_day)
        for day_idx in range(current_day_idx + 1, len(days)):
            if assigned_resources >= required_res:
                break

            current_day = days[day_idx]
            available = capacity.get((assigned_therapist, current_day), 0)

            if available >= capacity_multiplier:
                result_dict[(p, assigned_therapist, current_day, 1)] = 1
                capacity[(assigned_therapist, current_day)] -= capacity_multiplier
                therapist_load[assigned_therapist] += capacity_multiplier
                assigned_resources += 1
                print(f"  ✓ Session {assigned_resources}: T{assigned_therapist}, Day {current_day}")

        if assigned_resources >= required_res:
            print(f"  ✅ Patient {p} fully scheduled ({assigned_resources}/{required_res})\n")
            scheduled_patients.append(p)
        else:
            print(f"  ⚠️  Patient {p} partially scheduled ({assigned_resources}/{required_res})\n")
            unscheduled_patients.append(p)

    # ============================================================================
    # PASS 2: CONFLICT RESOLUTION FOR UNSCHEDULED PATIENTS
    # ============================================================================
    if unscheduled_patients:
        print("\n" + "=" * 100)
        print(f"PASS 2: Resolving {len(unscheduled_patients)} Unscheduled Patients".center(100))
        print("=" * 100 + "\n")

        resolved = []

        for p in unscheduled_patients:
            entry_day = entry_days[p]
            capacity_multiplier = capacity_multipliers[p]
            required_res = required_resources[p]

            print(f"Resolving Patient {p} (Entry: {entry_day}, Mult: {capacity_multiplier})")

            # === STRATEGY 1: Find a patient to swap ===
            # Look for scheduled patients on same entry day with >= capacity multiplier
            candidates_to_swap = []
            for other_p in scheduled_patients:
                if other_p == p:
                    continue
                if entry_days[other_p] != entry_day:
                    continue
                if capacity_multipliers[other_p] < capacity_multiplier:
                    continue

                # This patient could be swapped
                for t in therapists:
                    if result_dict.get((other_p, t, entry_day, 1), 0) == 1:
                        candidates_to_swap.append((capacity_multipliers[other_p], other_p, t))

            if candidates_to_swap:
                # Swap with smallest suitable patient
                candidates_to_swap.sort()
                _, swap_p, swap_t = candidates_to_swap[0]

                print(f"  ↔️  Swapping with Patient {swap_p} (T{swap_t})")

                # Remove swap_p from entry day
                result_dict[(swap_p, swap_t, entry_day, 1)] = 0
                capacity[(swap_t, entry_day)] += capacity_multipliers[swap_p]
                therapist_load[swap_t] -= capacity_multipliers[swap_p]

                # Assign p to entry day
                result_dict[(p, swap_t, entry_day, 1)] = 1
                capacity[(swap_t, entry_day)] -= capacity_multiplier
                therapist_load[swap_t] += capacity_multiplier

                print(f"  ✓ Patient {p} assigned to entry day")

                # Try to reschedule swap_p to later day
                current_day_idx = days.index(entry_day) + 1
                rescheduled = False

                for day_idx in range(current_day_idx, len(days)):
                    reschedule_day = days[day_idx]
                    if capacity.get((swap_t, reschedule_day), 0) >= capacity_multipliers[swap_p]:
                        result_dict[(swap_p, swap_t, reschedule_day, 1)] = 1
                        capacity[(swap_t, reschedule_day)] -= capacity_multipliers[swap_p]
                        therapist_load[swap_t] += capacity_multipliers[swap_p]
                        print(f"  ✓ Patient {swap_p} rescheduled to day {reschedule_day}")
                        rescheduled = True
                        break

                if not rescheduled:
                    print(f"  ⚠️  Could not reschedule patient {swap_p}")

                # Try to complete patient p
                assigned_resources = 1
                current_day_idx = days.index(entry_day) + 1
                for day_idx in range(current_day_idx, len(days)):
                    if assigned_resources >= required_res:
                        break
                    current_day = days[day_idx]
                    if capacity.get((swap_t, current_day), 0) >= capacity_multiplier:
                        result_dict[(p, swap_t, current_day, 1)] = 1
                        capacity[(swap_t, current_day)] -= capacity_multiplier
                        therapist_load[swap_t] += capacity_multiplier
                        assigned_resources += 1

                if assigned_resources >= required_res:
                    print(f"  ✅ Patient {p} fully resolved ({assigned_resources}/{required_res})\n")
                    resolved.append(p)
                else:
                    print(f"  ⚠️  Patient {p} still incomplete ({assigned_resources}/{required_res})\n")
            else:
                print(f"  ❌ No suitable swap candidate found for patient {p}\n")

        # Update unscheduled list
        still_unscheduled = [p for p in unscheduled_patients if p not in resolved]

        if still_unscheduled:
            print("\n" + "=" * 100)
            print(f"⚠️  WARNING: {len(still_unscheduled)} patients remain unscheduled".center(100))
            print("=" * 100)
            for p in still_unscheduled:
                print(f"  Patient {p} (Entry: {entry_days[p]}, Mult: {capacity_multipliers[p]})")
            print("=" * 100 + "\n")

    # ============================================================================
    # COMPUTE AUXILIARY DICTIONARIES
    # ============================================================================
    print("=" * 100)
    print("Computing Auxiliary Variables".center(100))
    print("=" * 100 + "\n")

    # Completion indicators and LOS
    completion_indicators = {}
    length_of_stay = {}
    max_day = max(days) if days else 0

    for p in patients:
        completion_day_found = False
        for d in days:
            completion_indicators[(p, d)] = 0
            total_resources_up_to_d = sum(
                result_dict.get((p, t, day, 1), 0)
                for t in therapists
                for day in days
                if days.index(day) <= days.index(d)
            )
            if not completion_day_found and total_resources_up_to_d >= required_resources[p]:
                completion_indicators[(p, d)] = 1
                completion_day_found = True

        entry_day = entry_days[p]
        completion_day = None
        for d in days:
            if completion_indicators[(p, d)] == 1:
                completion_day = d
                break

        if completion_day is not None:
            length_of_stay[p] = completion_day - entry_day + 1
        else:
            length_of_stay[p] = max_day + 1 - entry_day

        print(f"Patient {p}: Entry={entry_day}, Completion={completion_day}, LOS={length_of_stay[p]}")

    # y: No therapy session on day d
    y = {}
    for p in patients:
        entry_day = entry_days[p]
        discharge_day = None
        for d in days:
            if completion_indicators[(p, d)] == 1:
                discharge_day = d
                break

        for d in days:
            y[(p, d)] = 0
            if discharge_day is not None and entry_day <= d <= discharge_day:
                has_session = any(result_dict.get((p, t, d, 1), 0) == 1 for t in therapists)
                if not has_session:
                    y[(p, d)] = 1

    # z: Patient-therapist assignment
    z = {}
    for p in patients:
        for t in therapists:
            z[(p, t)] = 0
            if any(result_dict.get((p, t, d, 1), 0) == 1 for d in days):
                z[(p, t)] = 1

    # App: All zeros (placeholder for AI efficiency)
    App = {(p, d): 0 for p in patients for d in days}

    # S: Cumulative AI sessions
    S = {}
    for p in patients:
        discharge_day = None
        for d in days:
            if completion_indicators[(p, d)] == 1:
                discharge_day = d
                break

        cumulative_sum = 0
        for d in days:
            if discharge_day is not None and d > discharge_day:
                S[(p, d)] = 0
            else:
                cumulative_sum += y.get((p, d), 0)
                S[(p, d)] = cumulative_sum

    # l: Discharge indicator
    l = {}
    for p in patients:
        for d in days:
            l[(p, d)] = 0
            if completion_indicators.get((p, d), 0) == 1:
                l[(p, d)] = 1

    # Remaining capacity
    remaining_capacity_td = {(t, d): capacity.get((t, d), 0) for t in therapists for d in days if d > 0}
    remaining_capacity_d = {d: sum(capacity.get((t, d), 0) for t in therapists) for d in days if d > 0}

    print("\n" + "=" * 100)
    print("✅ INITIAL COLUMN GENERATION COMPLETE".center(100))
    print("=" * 100 + "\n")

    return result_dict, length_of_stay, y, z, App, S, l, remaining_capacity_td, remaining_capacity_d