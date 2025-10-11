import math


def check_feasibility_pre(x_dict11, y_dict11, P_Pre, df, p, R_p, Entry_p, app_data, M_p, Max_t,
                      learn='lin', deviation=None, verbose=False):
    """
    Checks if a given solution is feasible.

    Args:
        x_dict: Dictionary of the form {(p,t,d,iter): value, ...}
        y_dict: Dictionary of the form {(p,d): value, ...}
        P_Focus: Set of focus patients
        P_Post: Set of post patients
        df: DataFrame with the data
        p: Patient ID
        iteration: Current iteration
        R_p: Therapy requirements per patient
        Entry_p: Entry data per patient
        app_data: App efficiency data
        M_p: Big-M values per patient
        W_coeff: W coefficient
        Max_t: Dictionary with keys (t,d) indicating therapist availability
        learn: Learning curve type ('lin', 'exp', 'sigmoid' or value)
        reduction: Boolean for reduced subproblems
        deviation: Boolean for deviation constraints
        verbose: Boolean to control detailed daily printing

    Returns:
        (is_feasible, violations): Tuple with feasibility status and list of violations
    """

    index = p
    iteration = 1

    # Extract data from DataFrame
    D_raw = df['D_Full'].dropna().astype(int).unique().tolist()
    T = df['T'].dropna().astype(int).unique().tolist()


    x_dict1 = {}
    for t in T:
        for d in D_raw:
            key = (index, t, d)
            x_dict1[key] = x_dict11.get(key, 0)

    y_dict1 = {}
    for d in D_raw:
        key = (index, d)
        y_dict1[key] = y_dict11.get(key, 0)

    x_dict = {k: v for k, v in {(index, t, d, 1): value for (index, t, d), value in x_dict1.items()}.items() if k[0] == index}
    y_dict = {k: v for k, v in {(index, d, 1): value for (index, d), value in y_dict1.items()}.items() if k[0] == index}

    # Initialize data structures
    violations = []
    itr = iteration

    # Extract data from DataFrame
    D_raw = df['D_Full'].dropna().astype(int).unique().tolist()
    T = df['T'].dropna().astype(int).unique().tolist()

    D = D_raw

    # App efficiency parameters
    W = app_data["MS"][0]
    W_min = app_data["MS_min"][0]
    learn_type = app_data["learn_type"][0] if "learn_type" in app_data else learn
    base = app_data["theta_base"][0]
    lin_increase = app_data["lin_increase"][0]
    b = app_data["k_learn"][0]
    infl_point = app_data["infl_point"][0]

    # Calculate other variables based on x and y

    # 1. Calculate z (therapist assignment)
    z = {}
    for t in T:
        z[(p, t)] = 0
        for d in D:
            if (p, t, d, itr) in x_dict and x_dict[(p, t, d, itr)] > 0.5:
                z[(p, t)] = 1
                break

    print('Z', z)
    # 2. Calculate w (presence)
    w = {}
    for d in D:
        if d < Entry_p[p]:
            w[(p, d)] = 0
        elif d == Entry_p[p]:
            w[(p, d)] = 1
        else:
            # w[p,d] = 1 - sum(l[p,k] for k in range(Entry_p[p], d))
            # Since we don't have l directly, calculate w from the assignment constraint
            x_sum = sum(x_dict.get((p, t, d, itr), 0) for t in T)
            y_val = y_dict.get((p, d, itr), 0)
            w[(p, d)] = x_sum + y_val

    # 3. Calculate l (discharge indicator)
    l = {}
    active_days = [d for d in D if w.get((p, d), 0) == 1]
    if active_days:
        last_d = max(active_days)
    else:
        last_d = None

    for d in D:
        l[(p, d)] = 1 if d == last_d else 0

    print('L', l)
    print('W', w)

    # 4. Calculate LOS (Length of Stay)
    discharge_day = None
    for d in D:
        if l[(p, d)] > 0.5:
            discharge_day = d
            break



    LOS = discharge_day - Entry_p[p] + 1
    print('LOS', discharge_day, Entry_p[p], LOS)




    # 5. Calculate S (number of app appointments up to day d)
    S = {}
    for d in D:
        S[(p, d)] = sum(y_dict.get((p, t, itr), 0) for t in range(Entry_p[p], d + 1))

    # 6. Calculate app efficiency based on learn parameter
    App = {}
    if learn_type == 'lin' or learn == 'lin':
        for d in D:
            App[(p, d)] = min(1.0, base + lin_increase * S[(p, d)])
    elif learn_type == 'exp' or learn == 'exp':
        for d in D:
            s_val = S[(p, d)]
            App[(p, d)] = base + (1 - base) * (1 - math.exp(-b * s_val))
    elif learn_type == 'sigmoid' or learn == 'sigmoid':
        for d in D:
            s_val = S[(p, d)]
            App[(p, d)] = base + (1 - base) / (1 + math.exp(-b * (s_val - infl_point)))
    else:
        # Constant efficiency
        try:
            constant_eff = float(learn)
            for d in D:
                App[(p, d)] = constant_eff
        except:
            for d in D:
                App[(p, d)] = learn_type

    # 7. Calculate d_plus (deviation)
    d_plus = max(0, LOS - M_p[p]) if deviation else 0

    # Daily printing of schedule (only if verbose is True)
    if verbose:
        print(f"\n--- Daily Schedule for Patient {p} ---")
        for d in sorted(D):
            if d >= Entry_p[p] and w.get((p, d), 0) > 0:
                # Check if app session (y) or therapist session (x)
                app_session = y_dict.get((p, d, itr), 0) > 0.5
                therapist_session = any(x_dict.get((p, t, d, itr), 0) > 0.5 for t in T)

                # Determine session type
                if app_session:
                    session_type = "y"
                elif therapist_session:
                    session_type = "x"
                else:
                    session_type = "-"

                # Calculate current treatment equivalent up to this day
                therapy_sessions = sum(sum(x_dict.get((p, t, j, itr), 0) for j in range(Entry_p[p], d + 1)) for t in T)
                app_contribution = sum(y_dict.get((p, j, itr), 0) * App[(p, j)] for j in range(Entry_p[p], d + 1))
                current_equivalent = therapy_sessions + app_contribution

                print(f"Day {d}: ({session_type}) | Current Treatment Equivalent: {current_equivalent:.2f}")

    # Now check all constraints

    # Constraint: Discharge constraint
    l_sum = sum(l[(p, d)] for d in D)
    if p in P_Pre:
        if abs(l_sum - 1) > 1e-6:
            violations.append(f"Discharge constraint violated for focus patient {p}: sum of l = {l_sum}, should be 1")
    else:
        if l_sum > 1 + 1e-6:
            violations.append(f"No discharge constraint violated for patient {p}: sum of l = {l_sum}, should be <= 1")

    # Constraint: Single therapist
    z_sum = sum(z[(p, t)] for t in T)
    if abs(z_sum - 1) > 1e-6:
        violations.append(f"Single therapist constraint violated for patient {p}: sum of z = {z_sum}, should be 1")

    # Constraint: First day
    first_day_sum = sum(x_dict.get((p, t, Entry_p[p], itr), 0) for t in T)
    if abs(first_day_sum - 1) > 1e-6:
        violations.append(
            f"First day constraint violated for patient {p}: sum on day {Entry_p[p]} = {first_day_sum}, should be 1")

    # Constraint: Assignment constraint (w[p,d] = sum(x[p,t,d,itr]) + y[p,d])
    for d in D:
        x_sum = sum(x_dict.get((p, t, d, itr), 0) for t in T)
        y_val = y_dict.get((p, d, itr), 0)
        w_val = w[(p, d)]
        if abs(w_val - (x_sum + y_val)) > 1e-6:
            violations.append(
                f"Assignment constraint violated for patient {p}, day {d}: w = {w_val}, x_sum + y = {x_sum + y_val}")

    # Constraint: Therapist availability (x[p,t,d,itr] can only be > 0 if Max_t[t,d] allows it)
    for d in D:
        for t in T:
            x_val = x_dict.get((p, t, d, itr), 0)
            if x_val > 1e-6:  # If x is assigned (> 0)
                if (t, d) not in Max_t or Max_t[(t, d)] < 1e-6:  # If therapist not available on that day
                    violations.append(
                        f"Therapist availability violated for patient {p}, therapist {t}, day {d}: x = {x_val} but therapist not available (Max_t = {Max_t.get((t, d), 0)})")

    # Constraint: Therapist consistency (x[p,t,d,itr] <= z[p,t])
    for d in D:
        for t in T:
            x_val = x_dict.get((p, t, d, itr), 0)
            z_val = z[(p, t)]
            if x_val > z_val + 1e-6:
                violations.append(
                    f"Therapist consistency violated for patient {p}, therapist {t}, day {d}: x = {x_val} > z = {z_val}")

    # Constraint: Last day constraint (l[p,d] <= sum(x[p,t,d,itr]))
    for d in D:
        l_val = l[(p, d)]
        x_sum = sum(x_dict.get((p, t, d, itr), 0) for t in T)
        if l_val > x_sum + 1e-6:
            violations.append(f"Last day constraint violated for patient {p}, day {d}: l = {l_val} > x_sum = {x_sum}")

    # Constraint: Minimum weekly sessions (W_min) - only whole windows up to discharge day
    # Find discharge day for this patient
    patient_discharge_day = None
    for d in D:
        if l[(p, d)] > 0.5:
            patient_discharge_day = d
            break

    # If no discharge day found (P_Post not in P_Focus), use last active day
    if patient_discharge_day is None:
        active_days = [d for d in D if w.get((p, d), 0) > 0]
        if active_days:
            patient_discharge_day = max(active_days)
        else:
            patient_discharge_day = max(D)

    # Check only whole windows from entry day to discharge day
    window_start = Entry_p[p]
    while window_start + W - 1 <= patient_discharge_day:
        window_end = window_start + W - 1

        x_sum_week = sum(x_dict.get((p, t, j, itr), 0) for t in T for j in range(window_start, window_end + 1))
        w_sum_week = sum(w[(p, j)] for j in range(window_start, window_end + 1))
        required_min = W_min - W_min * (W - w_sum_week)

        print(f'Checking window {window_start}-{window_end}: with x_sum = {x_sum_week}, w_sum = {w_sum_week} and required_min = {required_min}')

        if x_sum_week < required_min - 1e-6:
            violations.append(
                f"Minimum weekly sessions violated for patient {p}, window [{window_start}, {window_end}]: x_sum = {x_sum_week} < required = {required_min}")

        window_start += 1  # Move to next non-overlapping window

    # Constraint: Equivalent constraint (R_p requirement)
    total_contribution = 0  # Initialize for use in summary

    # For P_Focus patients, check at discharge day
    if p in P_Pre:
        for d in D:
            if l[(p, d)] > 0.5:  # If patient is discharged on day d
                therapy_sessions = sum(sum(x_dict.get((p, t, j, itr), 0) for j in range(Entry_p[p], d + 1)) for t in T)

                if learn == 'lin' or learn_type == 'lin':
                    app_contribution = sum(y_dict.get((p, j, itr), 0) * App[(p, j)] for j in range(Entry_p[p], d + 1))
                else:
                    app_contribution = sum(y_dict.get((p, j, itr), 0) * App[(p, j)] for j in range(Entry_p[p], d + 1))

                total_contribution = therapy_sessions + app_contribution

                if total_contribution < R_p[p] - 1e-6:
                    violations.append(
                        f"Equivalent constraint violated for patient {p}, discharge day {d}: total = {total_contribution} < required = {R_p[p]}")

    # Constraint: Deviation constraints (if activated)
    if deviation:
        if d_plus < LOS - M_p[p] - 1e-6:
            violations.append(
                f"Deviation constraint 1 violated for patient {p}: d_plus = {d_plus} < LOS - M_p = {LOS - M_p[p]}")
        if d_plus < -1e-6:
            violations.append(f"Deviation constraint 2 violated for patient {p}: d_plus = {d_plus} < 0")

    # Check variable bounds
    for key, val in x_dict.items():
        if val < -1e-6 or val > 1 + 1e-6:
            violations.append(f"Variable bound violated for x{key}: value = {val}")

    for key, val in y_dict.items():
        if val < -1e-6 or val > 1 + 1e-6:
            violations.append(f"Variable bound violated for y{key}: value = {val}")

    # Summary
    is_feasible = len(violations) == 0

    if is_feasible:
        print(f"✓ Solution is feasible for patient {p} with Entry {Entry_p[p]} and Req: {R_p[p]}")
        print(f"  LOS: {LOS}")
        print(f"  Total therapy sessions: {sum(sum(x_dict.get((p, t, d, itr), 0) for t in T) for d in D)}")
        print(f"  Total app sessions: {sum(y_dict.get((p, d, itr), 0) for d in D)}")
        print(f"  Total contribution: {total_contribution}")
        if deviation:
            print(f"  Deviation (d_plus): {d_plus}")
    else:
        print(f"✗ Solution is NOT feasible for patient {p}")
        print(f"  Found {len(violations)} constraint violations:")
        for i, violation in enumerate(violations, 1):
            print(f"    {i}. {violation}")

    return is_feasible, violations


