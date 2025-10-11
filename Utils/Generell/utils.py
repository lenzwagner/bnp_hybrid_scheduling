import numpy as np


def boxed_print(*args, width=150, border='*', center=True):
    """
    Prints any number of inputs in a fixed-width box with borders.
    Automatically stringifies and wraps long lines.
    """
    text = '\n'.join(str(arg) for arg in args)
    lines = text.split('\n')

    print(border * (width + 2))
    for line in lines:
        while len(line) > width:
            part = line[:width]
            print(f"{border}{part.center(width) if center else part.ljust(width)}{border}")
            line = line[width:]
        print(f"{border}{line.center(width) if center else line.ljust(width)}{border}")
    #print(border * (width + 2))

def calc_appUsage(app_dict, los_dict, agg_to_patient):
    """
    Calculates the ratio between aggregated values from app_dict and los_dict
    for each unique first element of the tuple keys. Also returns the mean
    of these ratios.

    Parameters:
    app_dict (dict): Dictionary with tuple keys and numeric values.
    los_dict (dict): Dictionary with tuple keys and numeric values.

    Returns:
    tuple: (result_dict, mean_value)
        - result_dict: Dictionary containing the ratio for each unique first key.
        - mean_value: Mean of all calculated ratios.
    """
    # Extract unique first elements from the keys of both dictionaries
    first_keys_app_dict = {k[0] for k in app_dict.keys()}
    first_keys_los_dict = {k[0] for k in los_dict.keys()}

    # Calculate the sum of values for each unique first key in app_dict
    aggregated_app_dict = {
        key: sum(value for k, value in app_dict.items() if k[0] == key)
        for key in first_keys_app_dict
    }

    # Calculate the sum of values for each unique first key in los_dict
    aggregated_los_dict = {
        key: sum(value for k, value in los_dict.items() if k[0] == key)
        for key in first_keys_los_dict
    }

    # Calculate the ratio of aggregated values from app_dict and los_dict
    ratio_dict = {
        key: round(aggregated_app_dict[key] / aggregated_los_dict[key], 3)
        for key in aggregated_app_dict.keys()
    }

    add_ratio_dict = {patient_id: ratio_dict[profile_id] for patient_id, profile_id in agg_to_patient.items() if profile_id in ratio_dict}

    # Calculate the mean of the ratios
    mean_ratio = sum(add_ratio_dict.values()) / len(add_ratio_dict)

    return add_ratio_dict, round(mean_ratio, 3)

from collections import defaultdict


def sum_over_p(dicts):
    result = defaultdict(float)

    for d in dicts:
        for (p, t, d_key), value in d.items():
            result[(t, d_key)] += value

    return dict(result)


def calc_util(x_dict_full, Max_t_dict, D, c_g_j):
    from collections import defaultdict
    max_t_focus = {key: value for key, value in Max_t_dict.items() if 0 < key[1] <= max(D)}
    result = defaultdict(float)

    for d in x_dict_full:
        for (p, t, d_key), value in d.items():
            result[(t, d_key)] += value
    x_dict = dict(result)
    truncated_x_dict = {key: value for key, value in x_dict.items() if 0 < key[1] <= max(D)}

    keys_to_delete = [k for k, v in truncated_x_dict.items() if v == 0]
    dict1_cleaned = {k: v for k, v in truncated_x_dict.items() if k not in keys_to_delete}
    dict2_cleaned = {k: v for k, v in max_t_focus.items() if k not in keys_to_delete}

    ratio_dict = {
        key: round(dict1_cleaned[key] / (dict2_cleaned[key] or 1.0), 3)
        for key in dict1_cleaned.keys()
    }

    grouped = {}
    for (k1, k2), v in ratio_dict.items():
        if k1 not in grouped:
            grouped[k1] = []
        grouped[k1].append(v)

    means = {k: sum(vs) / len(vs) for k, vs in grouped.items()}

    extended_means = means.copy()
    next_key = max(means.keys()) + 1 if means else 1

    for profile_key, therapist_count in c_g_j.items():
        if profile_key in means and therapist_count > 1:
            for _ in range(therapist_count - 1):
                extended_means[next_key] = means[profile_key]
                next_key += 1

    average_of_means = sum(extended_means.values()) / len(extended_means) if extended_means else 0

    return extended_means, round(average_of_means, 3)


def app_eff_mapping(raw, D):
    out = defaultdict(list)
    for (p, t, d), val in raw.items():
        if val > 1e-9:  # > 0
            out[p].append(round(val, 2))

    out = {p: lst for p, lst in out.items()}
    for p in out:
        out[p] = out[p] + [np.nan] * (len(D) - len(out[p]))
    return out


def integrate_initial_solution_to_global(global_solutions, result_dict, length_of_stay, y, z, App, S, l,
                                         app_data, current_col, patients):
    """
    Integrates the dictionaries from initial_cg_starting_sol into the global solution.
    Creates a nested structure: {variable: {(p, solution_key): {original_keys: values}}}

    Parameters:
    - global_solutions: dict - The global solution dictionary
    - result_dict: dict - x dictionary from initial_cg_starting_sol
    - length_of_stay: dict - LOS dictionary from initial_cg_starting_sol
    - y: dict - y dictionary from initial_cg_starting_sol
    - z: dict - z dictionary from initial_cg_starting_sol
    - App: dict - App dictionary from initial_cg_starting_sol
    - S: dict - S dictionary from initial_cg_starting_sol
    - l: dict - l dictionary from initial_cg_starting_sol
    - app_data: dict - Contains E_app for determining var_names
    - current_col: int - Current column for solution_key
    - patients: list - List of patients

    Returns:
    - None (modifies global_solutions in-place)
    """

    # Bestimmen der var_names basierend auf E_app Struktur
    if isinstance(app_data["learn_type"][0], (int, float)):
        var_names = ['x', 'LOS', 'y', 'z', 'S', 'l']
    else:
        var_names = ['x', 'LOS', 'y', 'z', 'App', 'S', 'l']

    # Initialisiere global_solutions Dictionaries falls sie nicht existieren
    for var in var_names:
        if var not in global_solutions:
            global_solutions[var] = {}

    # Für jeden Patienten erstelle solution_key
    for p in patients:
        solution_key = (p, current_col)

        # x Dictionary - Sammle alle Einträge für diesen Patienten
        if 'x' in var_names:
            if solution_key not in global_solutions['x']:
                global_solutions['x'][solution_key] = {}
            for key, value in result_dict.items():
                if key[0] == p:  # key ist (p, t, d, 1), also Patient ist erste Position
                    global_solutions['x'][solution_key][key] = value

        # LOS Dictionary
        if 'LOS' in var_names:
            if solution_key not in global_solutions['LOS']:
                global_solutions['LOS'][solution_key] = {}
            if p in length_of_stay:
                extended_key = (p, 1)
                global_solutions['LOS'][solution_key][extended_key] = length_of_stay[p]

        # y Dictionary - Sammle alle Einträge für diesen Patienten
        if 'y' in var_names:
            if solution_key not in global_solutions['y']:
                global_solutions['y'][solution_key] = {}
            for (patient, d), value in y.items():
                if patient == p:
                    extended_key = (patient, d, 1)
                    global_solutions['y'][solution_key][extended_key] = value

        # z Dictionary - Sammle alle Einträge für diesen Patienten
        if 'z' in var_names:
            if solution_key not in global_solutions['z']:
                global_solutions['z'][solution_key] = {}
            for (patient, t), value in z.items():
                if patient == p:
                    extended_key = (patient, t, 1)
                    global_solutions['z'][solution_key][extended_key] = value


        # App Dictionary - falls verwendet
        if 'App' in var_names:
            if solution_key not in global_solutions['App']:
                global_solutions['App'][solution_key] = {}
            for (patient, d), value in App.items():
                if patient == p:
                    extended_key = (patient, d, 1)
                    global_solutions['App'][solution_key][extended_key] = value

        # S Dictionary - Sammle alle Einträge für diesen Patienten
        if 'S' in var_names:
            if solution_key not in global_solutions['S']:
                global_solutions['S'][solution_key] = {}
            for (patient, d), value in S.items():
                if patient == p:
                    extended_key = (patient, d, 1)
                    global_solutions['S'][solution_key][extended_key] = value

        # l Dictionary - Sammle alle Einträge für diesen Patienten
        if 'l' in var_names:
            if solution_key not in global_solutions['l']:
                global_solutions['l'][solution_key] = {}
            for (patient, d), value in l.items():
                if patient == p:
                    extended_key = (patient, d, 1)
                    global_solutions['l'][solution_key][extended_key] = value

    print(f"Initial solution successfully integrated into global_solutions with {len(var_names)} variables.")
    return global_solutions