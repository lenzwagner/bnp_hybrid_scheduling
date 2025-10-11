def initial_cg_starting_sol(max_capacity, patients, days, therapists, required_resources, entry_days, pre_assignments,
                            capacity_multipliers, flexible_patients, M_p, therapist_to_type = None):
    """
    Creates a dictionary with keys (p, t, d, 1) for all combinations of
    p in patients, d in days, and t in therapists with value 0.
    Then assigns resources based on entry_days and required_resources.
    If patients cannot be scheduled on their entry day, attempts to
    reschedule already scheduled patients.
    For patients in flexible_patients, checks if they have a discharge day.

    Parameters:
    - max_capacity: dict with keys (t, d) - capacities per day and therapist
    - patients: list of patients
    - days: list of days
    - therapists: list of therapists
    - required_resources: dict with key (p) - number of required resources per patient
    - entry_days: dict with key (p) - entry day per patient
    - pre_assignments: dict with pre-assigned schedules
    - capacity_multipliers: dict with key (p) - capacity multiplier per patient
    - flexible_patients: list of patients to check for discharge day
    - M_p: dict with key (p) - target length of stay per patient

    Returns:
    - result_dict: dictionary with keys (p, t, d, 1) and assigned values
    - completion_indicators: dictionary with keys (p, d)
    - length_of_stay: dictionary with keys (p)
    - unscheduled_patients: list of patients not scheduled on their entry day
    - y: dictionary with keys (p, d) - 1 if patient p has no therapy session on day d between entry and discharge
    - z: dictionary with keys (p, t) - 1 if patient p gets therapy from therapist t
    - App: dictionary with keys (p, d) - all zeros
    - S: dictionary with keys (p, d) - cumulative sum of y up to day d for each patient p
    - l: dictionary with keys (p, d) - 1 on discharge day, 0 otherwise
    """
    result_dict = {}
    capacity_copy = max_capacity.copy()
    capacity = {(t, d): v for (t, d), v in capacity_copy.items() if d > 0}
    pre_assignments_filtered = {(p, t, d): v for (p, t, d), v in pre_assignments.items() if d > 0}

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

    # Initialize result_dict with zeros
    for p in patients:
        for t in therapists:
            for d in days:
                result_dict[(p, t, d, 1)] = 0

    unscheduled_patients = []
    sorted_patients = sorted(patients, key=lambda p: (capacity_multipliers[p], -entry_days[p]), reverse=True)

    # Process each patient in prioritized order
    for p in sorted_patients:
        entry_day = entry_days[p]
        required_res = required_resources[p]
        capacity_multiplier = capacity_multipliers[p]
        assigned_resources = 0

        print(f"Processing patient {p}: Entry day={entry_day}, Required resources={required_res}, Capacity multiplier={capacity_multiplier}")

        # Find therapist with sufficient capacity
        assigned_therapist = None
        max_cap = -1
        for t in therapists:
            if capacity.get((t, entry_day), 0) >= capacity_multiplier and capacity.get((t, entry_day), 0) > max_cap:
                assigned_therapist = t
                max_cap = capacity.get((t, entry_day), 0)

        if assigned_therapist is None:
            print(f"*** SCHEDULING ERROR: Patient {p} CANNOT be scheduled on their entry day {entry_day}! ***")
            print(f"*** Reason: ALL therapists have capacity < {capacity_multiplier} on day {entry_day} ***")
            unscheduled_patients.append(p)
            continue

        result_dict[(p, assigned_therapist, entry_day, 1)] = 1
        capacity[(assigned_therapist, entry_day)] -= capacity_multiplier
        assigned_resources += 1

        if capacity[(assigned_therapist, entry_day)] == 0:
            print(f"  *** CAPACITY EXHAUSTED: Therapist {assigned_therapist}, Day {entry_day} now has capacity = 0 ***")

        print(
            f"  Assigned: Therapist {assigned_therapist}, Day {entry_day} (capacity reduced by {capacity_multiplier})")

        # Assign additional resources on subsequent days
        current_day_idx = days.index(entry_day)
        for day_idx in range(current_day_idx + 1, len(days)):
            if assigned_resources >= required_res:
                break
            current_day = days[day_idx]
            if capacity.get((assigned_therapist, current_day), 0) >= capacity_multiplier:
                result_dict[(p, assigned_therapist, current_day, 1)] = 1
                capacity[(assigned_therapist, current_day)] -= capacity_multiplier
                assigned_resources += 1
                if capacity[(assigned_therapist, current_day)] == 0:
                    print(
                        f"  *** CAPACITY EXHAUSTED: Therapist {assigned_therapist}, Day {current_day} now has capacity = 0 ***")
                print(
                    f"  Assigned: Therapist {assigned_therapist}, Day {current_day} (capacity reduced by {capacity_multiplier})")
            else:
                print(
                    f"  WARNING: Insufficient capacity for patient {p}, Therapist {assigned_therapist} on day {current_day} (needed: {capacity_multiplier}, available: {capacity.get((assigned_therapist, current_day), 0)})")

        if assigned_resources < required_res:
            print(f"  WARNING: Patient {p} was only assigned {assigned_resources} of {required_res} resources")
        print(f"  Patient {p} completed: {assigned_resources}/{required_res} resources assigned\n")

    # Second pass: Reschedule assigned patients to accommodate unscheduled ones
    remaining_unscheduled = []
    for p in unscheduled_patients:
        entry_day = entry_days[p]
        capacity_multiplier = capacity_multipliers[p]
        required_res = required_resources[p]

        print(f"Attempting to schedule patient {p} on entry day {entry_day} by rescheduling another patient")

        # Find patients scheduled on the entry day
        candidates = []
        for other_p in patients:
            if other_p != p:
                for t in therapists:
                    if result_dict.get((other_p, t, entry_day, 1), 0) == 1:
                        candidates.append((other_p, t, capacity_multipliers[other_p]))

        # Sort candidates by capacity_multipliers (prefer lower values)
        candidates = sorted(candidates, key=lambda x: x[2])

        scheduled = False
        for other_p, t, _ in candidates:
            # Check if rescheduling this patient frees enough capacity
            if capacity_multipliers[other_p] >= capacity_multiplier:
                print(
                    f"  Rescheduling patient {other_p} from therapist {t}, day {entry_day} to make room for patient {p}")

                # Remove the assignment
                result_dict[(other_p, t, entry_day, 1)] = 0
                capacity[(t, entry_day)] += capacity_multipliers[other_p]

                # Assign patient p
                result_dict[(p, t, entry_day, 1)] = 1
                capacity[(t, entry_day)] -= capacity_multiplier
                print(
                    f"  Assigned: Patient {p}, Therapist {t}, Day {entry_day} (capacity reduced by {capacity_multiplier})")

                # Reschedule other_p to a later day
                other_assigned_resources = sum(result_dict[(other_p, t, d, 1)] for d in days for t in therapists)
                other_required_resources = required_resources[other_p]
                current_day_idx = days.index(entry_day) + 1
                while other_assigned_resources < other_required_resources and current_day_idx < len(days):
                    current_day = days[current_day_idx]
                    if capacity.get((t, current_day), 0) >= capacity_multipliers[other_p]:
                        result_dict[(other_p, t, current_day, 1)] = 1
                        capacity[(t, current_day)] -= capacity_multipliers[other_p]
                        other_assigned_resources += 1
                        print(
                            f"  Patient {other_p} reassigned: Therapist {t}, Day {current_day} (capacity reduced by {capacity_multipliers[other_p]})")
                    current_day_idx += 1

                if other_assigned_resources < other_required_resources:
                    print(
                        f"  WARNING: Patient {other_p} was only reassigned {other_assigned_resources} of {other_required_resources} resources")
                else:
                    print(f"  Patient {other_p} successfully rescheduled")

                scheduled = True
                break

        if not scheduled:
            print(f"*** ERROR: Patient {p} could not be scheduled as no suitable patient could be rescheduled ***")
            remaining_unscheduled.append(p)

    # Update completion_indicators and length_of_stay
    completion_indicators = {}
    for p in patients:
        completion_day_found = False
        for d in days:
            completion_indicators[(p, d)] = 0
            total_resources_up_to_d = 0
            for day in days:
                if days.index(day) <= days.index(d):
                    total_resources_up_to_d += sum(result_dict[(p, t, day, 1)] for t in therapists)
            if not completion_day_found and total_resources_up_to_d >= required_resources[p]:
                completion_indicators[(p, d)] = 1
                completion_day_found = True

    length_of_stay = {}
    max_day = max(days) if days else 0
    for p in patients:
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

    # Create additional dictionaries as requested

    # y_pd: 1 if patient p has no therapy session on day d between entry and discharge, 0 otherwise
    y = {}
    for p in patients:
        entry_day = entry_days[p]
        # Find discharge day
        discharge_day = None
        for d in days:
            if completion_indicators[(p, d)] == 1:
                discharge_day = d
                break

        for d in days:
            y[(p, d)] = 0
            # Check if day d is between entry and discharge
            if discharge_day is not None and entry_day <= d <= discharge_day:
                # Check if patient has therapy session on day d
                has_session = any(result_dict.get((p, t, d, 1), 0) == 1 for t in therapists)
                if not has_session:
                    y[(p, d)] = 1

    # z_pt: 1 if patient p gets therapy from therapist t, 0 otherwise
    z = {}
    for p in patients:
        for t in therapists:
            z[(p, t)] = 0
            # Check if patient p has any session with therapist t
            if any(result_dict.get((p, t, d, 1), 0) == 1 for d in days):
                z[(p, t)] = 1

    # App_pd: all zeros for every p and d
    App = {}
    for p in patients:
        for d in days:
            App[(p, d)] = 0

    # S_pd: cumulative sum of y up to day d for each patient p, but 0 after discharge day
    S = {}
    for p in patients:
        # Find discharge day
        discharge_day = None
        for d in days:
            if completion_indicators[(p, d)] == 1:
                discharge_day = d
                break

        cumulative_sum = 0
        for d in days:
            if discharge_day is not None and d > discharge_day:
                # After discharge day, S should be 0
                S[(p, d)] = 0
            else:
                cumulative_sum += y.get((p, d), 0)
                S[(p, d)] = cumulative_sum

    # l_pd: 1 on discharge day, 0 otherwise
    l = {}
    for p in patients:
        for d in days:
            l[(p, d)] = 0
            if completion_indicators.get((p, d), 0) == 1:
                # This is the discharge day
                l[(p, d)] = 1

    # Check discharge day for flexible patients
    for p in flexible_patients:
        completion_day = None
        for d in days:
            if completion_indicators[(p, d)] == 1:
                completion_day = d
                break
        if completion_day is not None:
            print(f"Flexible patient {p} has a discharge day on day {completion_day}")
        else:
            print(f"ERROR: Flexible patient {p} has NO discharge day")

    if remaining_unscheduled:
        print(
            f"\n*** CRITICAL ERROR: {len(remaining_unscheduled)} patient(s) could NOT be scheduled on their entry day: ***")
        for p in remaining_unscheduled:
            print(
                f"  - Patient {p} (Entry day: {entry_days[p]}, Required capacity: {capacity_multipliers[p]}, Available capacity: Therapist 1={capacity.get((1, entry_days[p]), 0)}, Therapist 2={capacity.get((2, entry_days[p]), 0)})")
    else:
        print("\n*** ALL patients were scheduled on their entry day! ***")


    print("\n*** REDUCED CAPACITY ***")

    remaining_capacity_td = {}
    for (t, d), cap in capacity.items():
        remaining_capacity_td[(t, d)] = cap
        print(f"Remaining capacity - Therapist {t}, Day {d}: {cap}")

    remaining_capacity_d = {}
    for d in days:
        if d > 0:  # nur für gültige Tage
            total_capacity_day = sum(capacity.get((t, d), 0) for t in therapists)
            remaining_capacity_d[d] = total_capacity_day
            print(f"Total remaining capacity - Day {d}: {total_capacity_day}")

    print("*** END CAPACITY OVERVIEW ***\n")

    if remaining_unscheduled:
        print(
            f"\n*** CRITICAL ERROR: {len(remaining_unscheduled)} patient(s) could NOT be scheduled on their entry day: ***")
        for p in remaining_unscheduled:
            print(
                f"  - Patient {p} (Entry day: {entry_days[p]}, Required capacity: {capacity_multipliers[p]}, Available capacity: Therapist 1={capacity.get((1, entry_days[p]), 0)}, Therapist 2={capacity.get((2, entry_days[p]), 0)})")
    else:
        print("\n*** ALL patients were scheduled on their entry day! ***")

    return result_dict, length_of_stay, y, z, App, S, l, remaining_capacity_td, remaining_capacity_d