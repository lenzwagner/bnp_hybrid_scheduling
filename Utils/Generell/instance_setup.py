import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
from Utils.Generell.utils import *


def compare_patient_solutions(start_x, start_y, start_LOS, P, T, D):
    """
    Compares solutions for each patient, including appointment treatments, and groups identical solutions.

    Args:
        start_x: Dictionary with (p, t, d) keys for treatment assignments
        start_y: Dictionary with (p, d) keys for appointment treatments
        start_LOS: Dictionary with LOS values per patient
        P: List of patients
        T: List of timeslots
        D: List of days

    Returns:
        Dictionary with groups of identical patients
    """
    patient_patterns = {}
    for p in P:
        x_values = {(t, d): start_x.get((p, t, d), 0) for t in T for d in D}
        y_values = {d: start_y.get((p, d), 0) for d in D}
        los_value = start_LOS.get(p, 0)

        pattern_key = (
            tuple((k, v) for k, v in sorted(x_values.items()) if v > 0.5),
            tuple((k, v) for k, v in sorted(y_values.items()) if v > 0.5),
            los_value
        )

        if pattern_key in patient_patterns:
            patient_patterns[pattern_key].append(p)
        else:
            patient_patterns[pattern_key] = [p]

    result = {}
    for i, (pattern, patients) in enumerate(patient_patterns.items(), 1):
        result[f"Group_{i}"] = {
            "Patients": patients,
            "Count": len(patients),
            "Treatment_Days": pattern[0],
            "Appointment_Days": pattern[1],
            "LOS": pattern[2]
        }

    return result


def print_comparison_results(results):
    """
    Prints details of grouped patient results.

    Displays information about patient groups, including patient lists,
    counts, treatment days, appointment days, and length of stay.

    Args:
        results (dict): Dictionary with grouped patient information
    """
    print("\nGrouping of Identical Patients:")
    print(f"Groups Found: {len(results)}")
    for gruppe, details in results.items():
        print(f"\n{gruppe}:")
        print(f"  Patients: {details['Patienten']}")
        print(f"  Count: {details['Anzahl']}")
        print(f"  Treatment Days (t,d): {details['Behandlungstage']}")
        print(f"  Appointment Days (d): {details['Appointment_Tage']}")
        print(f"  LOS: {details['LOS']}")


def get_unique_combinations_and_list_with_dicts(R_p, Entry_p, P, filter_entry_ge_1=False):
    """
    Identifies unique patient combinations based on stay duration and entry date.

    Groups patients with identical stay duration and entry dates, creating
    profiles for further analysis and scheduling optimization. Optionally filters
    patients with entry date >= 1.

    Args:
        R_p (dict): Patient stay duration dictionary
        Entry_p (dict): Patient entry date dictionary
        P (list): List of patient IDs to consider
        filter_entry_ge_1 (bool): If True, only consider patients with entry date >= 1

    Returns:
        Tuple containing:
        - N_c: List of unique patient profile indices
        - R_p_c: Stay duration for each unique profile
        - Entry_p_c: Entry date for each unique profile
        - Profile_patient_count: Number of patients per profile
        - unique_combinations: Mapping of unique combinations to patient lists
        - num_unique_combinations: Total number of unique patient profiles
        - patient_to_profile_mapping: Dictionary mapping each original patient to their profile
    """
    # Kombinationen erstellen, ggf. mit Filter für Entry_p >= 1
    valid_patients = []
    patient_combinations = {}

    for p in P:
        if p in R_p and p in Entry_p:
            if filter_entry_ge_1 and Entry_p[p] < 1:
                continue
            combo = (R_p[p], Entry_p[p])
            valid_patients.append(p)
            patient_combinations[p] = combo

    # Dictionary für einzigartige Kombinationen
    unique_combinations = {}
    for p in valid_patients:
        combo = patient_combinations[p]
        if combo not in unique_combinations:
            unique_combinations[combo] = []
        unique_combinations[combo].append(p)

    num_unique_combinations = len(unique_combinations)

    N_c = list(range(1, num_unique_combinations + 1))

    R_p_c = {}
    Entry_p_c = {}
    Profile_patient_count = {}
    patient_to_profile_mapping = {}

    for idx, (combo, patients) in enumerate(unique_combinations.items(), start=1):
        R_p_c[idx] = combo[0]  # Stay Duration
        Entry_p_c[idx] = combo[1]  # Entry Date
        Profile_patient_count[idx] = len(patients)

        # Mapping für jeden ursprünglichen Patienten zu seinem Profil
        for patient in patients:
            patient_to_profile_mapping[patient] = idx

    # Ausgabe, welche Patienten kombiniert wurden
    print("\nUnique patient groupings:")
    for idx, (combo, patients) in enumerate(unique_combinations.items(), start=1):
        print(f"Group {idx}: Stay Duration = {combo[0]}, Entry Date = {combo[1]}, Patients = {patients}")

    print(f"\nPatient to Profile Mapping:")
    for patient, profile in patient_to_profile_mapping.items():
        print(f"Patient {patient} -> Profile {profile}")

    return N_c, R_p_c, Entry_p_c, Profile_patient_count, patient_to_profile_mapping

def categorize_patients_full(Entry_p, D):
    """
    Kategorisiert Patienten in Pre-, Focus- und Post-Gruppen.
    Args:
        Entry_p (dict): Eintrittstage der Patienten
        D (list): Fokus-Horizont
        D_full (list): Voller Planungshorizont
    Returns:
        tuple: (P_Pre, P_F, P_Post)
    """

    P_Pre = [p for p, d in Entry_p.items() if d < min(D)]
    P_F = [p for p, d in Entry_p.items() if min(D) <= d <= max(D)]
    P_Post = [p for p, d in Entry_p.items() if d > max(D)]
    P_Join = sorted(P_F + P_Post)
    E_dict = {item: 1 if item in P_F else 0 for item in P_Join}

    return P_Pre, P_F, P_Post, P_Join, E_dict



def generate_patient_data(T=10, D_focus=30, W_on=5, W_off=2, daily=4, pttr_scenario='medium', seed=42, plot_show=False):
    """
    Generate patient data based on computational study specifications.

    Parameters:
    - T: Number of therapists (default: 10)
    - D_focus: Focus horizon days (default: 30)
    - W_on: Working days per cycle (default: 5)
    - W_off: Off days per cycle (default: 2)
    - daily: Daily therapist capacity (default: 4)
    - pttr_scenario: 'light' (50%), 'medium' (70%), or 'heavy' (90%) utilization
    - seed: Random seed for reproducibility
    - plot_show: Whether to show plots

    Returns:
    - requirements_dict: Dictionary with patient requirements
    - entry_day_dict: Dictionary with patient entry days
    - Max_t: Dictionary with therapist capacity by day
    - P: List of patient IDs
    - D: List of focus days
    - D_planning: List of planning horizon days
    - D_full: List of full planning horizon days
    - T: List of therapist IDs
    - M_p: Dictionary with mean LOS by patient
    - W_coeff: Work cycle coefficient
    """

    np.random.seed(seed)
    random.seed(seed)

    # Calculate derived parameters based on study specifications
    W_coeff = W_on / (W_on + W_off)

    # DRG data from the study
    drg_data = {
        'E65A': {'los_min': 6, 'los_max': 30, 'los_mean': 17.9, 'cases': 6469, 'percentage': 4.8},
        'E65B': {'los_min': 3, 'los_max': 16, 'los_mean': 8.0, 'cases': 37616, 'percentage': 27.8},
        'E65C': {'los_min': 2, 'los_max': 12, 'los_mean': 6.1, 'cases': 91161, 'percentage': 67.4}
    }

    # Calculate maximum required treatments
    max_req = max(group['los_max'] for group in drg_data.values())
    D_max = int(np.ceil(max_req * 1 / W_coeff) + 2)

    # Define horizons according to study
    D = list(range(1, D_focus + 1))  # Focus horizon
    D_planning = list(range(1, D_focus + D_max + 1))  # Planning horizon
    D_ext = list(range(-D_max + 1, D_focus + D_max + 1))  # Extended horizon
    D_full = D_ext  # Full planning horizon same as extended

    # Calculate total capacity over extended horizon
    total_days = len(D_full)
    working_days = math.ceil(total_days * W_coeff)
    daily_capacity = daily
    total_capacity = working_days * daily_capacity * T
    print('FFFF', total_days, working_days, daily_capacity, total_capacity)

    # Define PTTR scenarios
    utilization_rates = {
        'light': 0.60,
        'medium': 0.75,
        'heavy': 0.90
    }

    if pttr_scenario not in utilization_rates:
        raise ValueError("pttr_scenario must be 'light', 'medium', or 'heavy'")

    utilization = utilization_rates[pttr_scenario]
    target_sessions = int(total_capacity * utilization)

    total_cases = sum(group['cases'] for group in drg_data.values())

    # Calculate average LOS
    avg_mean_unweighted_unrounded = sum(group['los_mean'] * group['percentage'] / 100 for group in drg_data.values())
    avg_mean_unweighted = math.ceil(avg_mean_unweighted_unrounded * (10 ** 1)) / (10 ** 1)

    # Calculate number of patients
    P_Nr = int(np.ceil(target_sessions / avg_mean_unweighted))

    # Generate therapist availability schedule
    Max_t = {}

    start_days = [1, 2, 3, 4, 5]
    therapist_start_days = {}

    for t in range(1, T + 1):
        therapist_start_days[t] = start_days[(t - 1) % len(start_days)]
        print(f'Start_day t{t}: {therapist_start_days[t]}')
        start_offset = therapist_start_days[t]  # e.g. d=1

        for d in D_full:

            day_of_week = (d - start_offset) % 7

            print(f'Day_of_week t{t} at d={d}: {day_of_week}')

            if 0 <= day_of_week < W_on:
                Max_t[(t, d)] = daily_capacity
            else:
                Max_t[(t, d)] = 0

    # Debug: Verify average number of therapists per day of the week
    day_counts = {d % 7: 0 for d in range(7)}
    for d in D_full:
        therapists_working = sum(1 for t in range(1, T + 1) if Max_t[(t, d)] > 0)
        day_counts[(d - 1) % 7] += therapists_working
    print("\n=== Therapist Availability by Day of Week ===")
    for dow in day_counts:
        avg_therapists = day_counts[dow] / (total_days // 7)
        print(f"Day {dow} (0=Monday, 5=Saturday, 6=Sunday): Avg therapists = {avg_therapists:.2f}")

    # Generate patient distribution based on DRG proportions
    patients_per_group = {}
    for drg in drg_data:
        proportion = drg_data[drg]['cases'] / total_cases
        patients_per_group[drg] = int(np.floor(P_Nr * proportion))

    # Handle remaining patients
    remaining_patients = P_Nr - sum(patients_per_group.values())
    if remaining_patients > 0:
        largest_group = max(drg_data.items(), key=lambda x: x[1]['cases'])[0]
        patients_per_group[largest_group] += remaining_patients

    def generate_bounded_lognormal(min_val, max_val, mean_val, size=1):
        """Generate log-normal distributed values within bounds"""
        sigma = 0.5
        mu = np.log(mean_val) - (sigma ** 2) / 2
        values = []
        while len(values) < size:
            sample = np.random.lognormal(mu, sigma)
            if min_val <= sample <= max_val:
                values.append(sample)
        return values[0] if size == 1 else values

    # Generate patients
    patients = []
    patient_id = 1

    for drg, num_patients in patients_per_group.items():
        for _ in range(num_patients):
            requirements = round(generate_bounded_lognormal(
                drg_data[drg]['los_min'],
                drg_data[drg]['los_max'],
                drg_data[drg]['los_mean']
            ))

            # Generate admission date with weekend (Saturday and Sunday) probability adjustment
            num_weekendD_ext = 0
            num_weekdayD_ext = 0

            for dayy in D_ext:
                day_of_week = (dayy - 1) % 7
                if day_of_week in {5, 6}:  # Saturday (5) and Sunday (6)
                    num_weekendD_ext += 1
                else:
                    num_weekdayD_ext += 1

            Z = num_weekdayD_ext + 0.5 * num_weekendD_ext

            probabilities = []
            for dayy in D_ext:
                day_of_week = (dayy - 1) % 7
                if day_of_week in {5, 6}:  # Saturday (5) and Sunday (6)
                    prob = 0.5 / Z
                else:
                    prob = 1 / Z
                probabilities.append(prob)

            day = random.choices(D_ext, weights=probabilities, k=1)[0]

            patients.append({
                'patient_id': patient_id,
                'drg_group': drg,
                'requirements': requirements,
                'entry_day': day
            })
            patient_id += 1

    # Create output dictionaries
    P = list(range(1, len(patients) + 1))
    M_p = {}
    for patient in patients:
        patient_id = patient['patient_id']
        drg_group = patient['drg_group']
        M_p[patient_id] = drg_data[drg_group]['los_mean']

    # Convert to DataFrames for analysis
    df = pd.DataFrame(patients)

    # Print summary statistics
    print(f"\n=== Patient Data Generation Summary ===")
    print(f"PTTR Scenario: {pttr_scenario.upper()} ({utilization * 100:.0f}% utilization)")
    print(f"Target Sessions: {target_sessions:,}")
    print(f"Generated Patients: {len(patients):,}")
    print(f"Patient-to-Therapist Ratio: {len(patients) / T:.1f}:1")
    print(f"Total Therapist Capacity: {total_capacity:,} sessions")
    print(f"Average LOS: {avg_mean_unweighted:.2f} sessions")

    print(f"\n=== Horizon Definitions ===")
    print(f"Focus Horizon (D): {min(D)} to {max(D)} ({len(D)} days)")
    print(f"Planning Horizon: {min(D_planning)} to {max(D_planning)} ({len(D_planning)} days)")
    print(f"Extended Horizon (D_ext): {min(D_ext)} to {max(D_ext)} ({len(D_ext)} days)")

    print(f"\n=== DRG Group Distribution ===")
    print(df['drg_group'].value_counts().sort_index())

    print(f"\n=== Requirements Statistics by DRG Group ===")
    for drg in ['E65A', 'E65B', 'E65C']:
        if drg in df['drg_group'].values:
            print(f"\n{drg}:")
            print(df[df['drg_group'] == drg]['requirements'].describe())

    # Generate plots if requested
    if plot_show:
        plt.figure(figsize=(20, 15))

        # DRG distribution
        plt.subplot(3, 2, 1)
        sns.countplot(data=df, x='drg_group', order=['E65A', 'E65B', 'E65C'])
        plt.title('Distribution of DRG Groups')
        plt.xlabel('DRG Group')
        plt.ylabel('Count')

        # Entry day distribution
        plt.subplot(3, 2, 2)
        day_bins = np.arange(min(D_ext), max(D_ext) + 2) - 0.5
        sns.histplot(data=df, x='entry_day', bins=day_bins, discrete=True, stat="density")
        plt.title(f'Density of Entry Days (Range: {min(D_ext)} to {max(D_ext)})')
        plt.xlabel('Day')
        plt.ylabel('Density')

        # Requirements by DRG group
        for idx, drg in enumerate(['E65A', 'E65B', 'E65C']):
            plt.subplot(3, 2, idx + 3)
            drg_data_filtered = df[df['drg_group'] == drg]
            if not drg_data_filtered.empty:
                sns.histplot(data=drg_data_filtered, x='requirements',
                             bins=range(df['requirements'].min(), df['requirements'].max() + 1),
                             discrete=True, stat="density")
            plt.title(f'Density of Requirements for {drg}')
            plt.xlabel('Number of Required Treatments')
            plt.ylabel('Density')

        # Overall requirements
        plt.subplot(3, 2, 6)
        sns.histplot(data=df, x='requirements',
                     bins=range(df['requirements'].min(), df['requirements'].max() + 1),
                     discrete=True, stat="density")
        plt.title('Overall Density of Requirements')
        plt.xlabel('Number of Required Treatments')
        plt.ylabel('Density')

        plt.tight_layout()
        plt.show()

    requirements_dict = df.set_index("patient_id")["requirements"].to_dict()
    entry_day_dict = df.set_index("patient_id")["entry_day"].to_dict()

    return requirements_dict, entry_day_dict, Max_t, P, D, D_planning, D_full, list(range(1, T + 1)), M_p, W_coeff

def aggregate_therapists(T, Max_t, W_on, W_off, D):
    # Create therapist types based on W_on, W_off, and capacity
    therapist_types = {}
    G_C = []
    g_j_C = {}  # Number of therapists per type
    Q_jd_Agg = {}  # Aggregated capacity per type and day
    therapist_to_type = {}  # Mapping of therapist to type

    print("Starting therapist aggregation...")
    for t in T:
        # Construct key
        key = (W_on, W_off, tuple(Max_t[t, d] for d in D))
        print(f"Therapist {t}: key = {key}")

        if key not in therapist_types:
            j = len(G_C) + 1
            therapist_types[key] = j
            G_C.append(j)
            g_j_C[j] = 0
            Q_jd_Agg[j] = {d: 0 for d in D}
            print(f"  --> New type created: Type {j} with key {key}")

        j = therapist_types[key]
        therapist_to_type[t] = j
        g_j_C[j] += 1

        for d in D:
            Q_jd_Agg[j][d] += Max_t[t, d]
            #print(f"  --> Added Max_t[{t},{d}] = {Max_t[t,d]} to Type {j}, total for day {d}: {Q_jd_Agg[j][d]}")

    Q_flat = {(t,d): Q_jd_Agg[t][d] for t in Q_jd_Agg for d in Q_jd_Agg[t]}

    print("\n=== Aggregation Results ===")
    print(f"G_C (therapist types): {G_C}")
    print(f"g_j_C (number per type): {g_j_C}")
    print(f"Q_flat (aggregated capacities): {Q_flat}")
    print(f"therapist_to_type mapping: {therapist_to_type}")
    print("===========================\n")

    return G_C, g_j_C, Q_flat, therapist_to_type


def generate_patient_data_log(T=10, D_focus=30, W_on=5, W_off=2, daily=4, pttr_scenario='medium', seed=42, plot_show=False):
    """
    Generate patient data based on computational study specifications.

    Parameters:
    - T: Number of therapists (default: 10)
    - D_focus: Focus horizon days (default: 30)
    - W_on: Working days per cycle (default: 5)
    - W_off: Off days per cycle (default: 2)
    - daily: Daily therapist capacity (default: 4)
    - pttr_scenario: 'light' (50%), 'medium' (70%), or 'heavy' (90%) utilization
    - seed: Random seed for reproducibility
    - plot_show: Whether to show plots

    Returns:
    - requirements_dict: Dictionary with patient requirements
    - entry_day_dict: Dictionary with patient entry days
    - Max_t: Dictionary with therapist capacity by day
    - P: List of patient IDs
    - D: List of focus days
    - D_planning: List of planning horizon days
    - D_full: List of full planning horizon days
    - T: List of therapist IDs
    - M_p: Dictionary with mean LOS by patient
    - W_coeff: Work cycle coefficient
    """

    np.random.seed(seed)
    random.seed(seed)

    # Calculate derived parameters based on study specifications
    W_coeff = W_on / (W_on + W_off)

    # DRG data from the study
    drg_data = {
        'E65A': {
            'los_min': 6, 'los_max': 30, 'los_mean': 17.9, 'cases': 6469, 'percentage': 4.8,
            'distribution': 'lognormal'
        },
        'E65B': {
            'los_min': 3, 'los_max': 16, 'los_mean': 8.0, 'cases': 37616, 'percentage': 27.8,
            'distribution': 'lognormal'
        },
        'E65C': {
            'los_min': 2, 'los_max': 12, 'los_mean': 6.1, 'cases': 91161, 'percentage': 67.4,
            'distribution': 'gamma'
        }
    }

    # Calculate distribution parameters based on the methodology described in the paper
    def calculate_distribution_parameters(drg_info):
        """Calculate distribution parameters based on min, max, and mean LOS"""
        x_min = drg_info['los_min']
        x_max = drg_info['los_max']
        x_mean = drg_info['los_mean']

        if drg_info['distribution'] == 'lognormal':
            # Log-Normal Parameter Estimation
            # σ_L ≈ (ln(x_max) - ln(x_min)) / 4
            sigma_L = (np.log(x_max) - np.log(x_min)) / 4

            # μ_L = ln(x̄) - σ_L² / 2
            mu_L = np.log(x_mean) - (sigma_L ** 2) / 2

            drg_info['mu_L'] = mu_L
            drg_info['sigma_L'] = sigma_L

        elif drg_info['distribution'] == 'gamma':
            # Gamma Parameter Estimation
            # σ ≈ (x_max - x_min) / 4
            sigma = (x_max - x_min) / 4

            # k = (x̄ / σ)²
            k = (x_mean / sigma) ** 2

            # θ = σ² / x̄
            theta = (sigma ** 2) / x_mean

            drg_info['k'] = k
            drg_info['theta'] = theta
            drg_info['sigma_approx'] = sigma

    # Calculate parameters for all DRG groups
    for drg in drg_data:
        calculate_distribution_parameters(drg_data[drg])

    # Calculate maximum required treatments
    max_req = max(group['los_max'] for group in drg_data.values())
    D_max = int(np.ceil(max_req * 1 / W_coeff) + 2)

    print('DDDDDD', D_max, W_coeff)

    # Define horizons according to study
    D = list(range(1, D_focus + 1))  # Focus horizon
    D_planning = list(range(1, D_focus + D_max + 1))  # Planning horizon
    D_ext = list(range(-D_max + 1, D_focus + D_max + 1))  # Extended horizon
    D_full = D_ext  # Full planning horizon same as extended

    # Calculate total capacity over extended horizon
    total_days = len(D_full)
    working_days = math.ceil(total_days * W_coeff)
    daily_capacity = daily
    total_capacity = working_days * daily_capacity * T
    print('FFFF', total_days, working_days, daily_capacity, total_capacity)


    # Define PTTR scenarios
    utilization_rates = {
        'light': 0.60,
        'medium': 0.75,
        'heavy': 0.90
    }

    if pttr_scenario not in utilization_rates:
        raise ValueError("pttr_scenario must be 'light', 'medium', or 'heavy'")

    utilization = utilization_rates[pttr_scenario]
    target_sessions = int(total_capacity * utilization)

    total_cases = sum(group['cases'] for group in drg_data.values())

    # Calculate average LOS
    avg_mean_unweighted_unrounded = sum(group['los_mean'] * group['percentage'] / 100 for group in drg_data.values())
    avg_mean_unweighted = math.ceil(avg_mean_unweighted_unrounded * (10 ** 1)) / (10 ** 1)

    # Calculate number of patients
    P_Nr = int(np.ceil(target_sessions / avg_mean_unweighted))

    # Generate therapist availability schedule
    Max_t = {}

    start_days = [1, 2, 3, 4, 5]
    therapist_start_days = {}

    for t in range(1, T + 1):
        therapist_start_days[t] = start_days[(t - 1) % len(start_days)]
        print(f'Start_day t{t}: {therapist_start_days[t]}')
        start_offset = therapist_start_days[t]  # e.g. d=1

        for d in D_full:

            day_of_week = (d - start_offset) % 7

            print(f'Day_of_week t{t} at d={d}: {day_of_week}')

            if 0 <= day_of_week < W_on:
                Max_t[(t, d)] = daily_capacity
            else:
                Max_t[(t, d)] = 0

    # Debug: Verify average number of therapists per day of the week
    day_counts = {d % 7: 0 for d in range(7)}
    for d in D_full:
        therapists_working = sum(1 for t in range(1, T + 1) if Max_t[(t, d)] > 0)
        day_counts[(d - 1) % 7] += therapists_working
    print("\n=== Therapist Availability by Day of Week ===")
    for dow in day_counts:
        avg_therapists = day_counts[dow] / (total_days // 7)
        print(f"Day {dow} (0=Monday, 5=Saturday, 6=Sunday): Avg therapists = {avg_therapists:.2f}")

    # Generate patient distribution based on DRG proportions
    patients_per_group = {}
    for drg in drg_data:
        proportion = drg_data[drg]['cases'] / total_cases
        patients_per_group[drg] = int(np.floor(P_Nr * proportion))

    # Handle remaining patients
    remaining_patients = P_Nr - sum(patients_per_group.values())
    if remaining_patients > 0:
        largest_group = max(drg_data.items(), key=lambda x: x[1]['cases'])[0]
        patients_per_group[largest_group] += remaining_patients

    def generate_los_sample(drg_group, drg_info):
        """Generate a single LOS sample based on the appropriate distribution"""
        if drg_info['distribution'] == 'lognormal':
            # Generate from log-normal distribution
            while True:
                sample = np.random.lognormal(drg_info['mu_L'], drg_info['sigma_L'])
                # Ensure sample is within bounds
                if drg_info['los_min'] <= sample <= drg_info['los_max']:
                    return round(sample)

        elif drg_info['distribution'] == 'gamma':
            # Generate from gamma distribution
            while True:
                sample = np.random.gamma(drg_info['k'], drg_info['theta'])
                # Ensure sample is within bounds
                if drg_info['los_min'] <= sample <= drg_info['los_max']:
                    return round(sample)

        else:
            raise ValueError(f"Unknown distribution type: {drg_info['distribution']}")

    # Generate patients
    patients = []
    patient_id = 1

    for drg, num_patients in patients_per_group.items():
        drg_info = drg_data[drg]
        for _ in range(num_patients):
            requirements = generate_los_sample(drg, drg_info)

            # Generate admission date with weekend (Saturday and Sunday) probability adjustment
            num_weekendD_ext = 0
            num_weekdayD_ext = 0

            for dayy in D_ext:
                day_of_week = (dayy - 1) % 7
                if day_of_week in {5, 6}:  # Saturday (5) and Sunday (6)
                    num_weekendD_ext += 1
                else:
                    num_weekdayD_ext += 1

            Z = num_weekdayD_ext + 0.5 * num_weekendD_ext

            probabilities = []
            for dayy in D_ext:
                day_of_week = (dayy - 1) % 7
                if day_of_week in {5, 6}:  # Saturday (5) and Sunday (6)
                    prob = 0.5 / Z
                else:
                    prob = 1 / Z
                probabilities.append(prob)

            day = random.choices(D_ext, weights=probabilities, k=1)[0]

            patients.append({
                'patient_id': patient_id,
                'drg_group': drg,
                'requirements': requirements,
                'entry_day': day
            })
            patient_id += 1

    # Create output dictionaries
    P = list(range(1, len(patients) + 1))
    M_p = {}
    for patient in patients:
        patient_id = patient['patient_id']
        drg_group = patient['drg_group']
        M_p[patient_id] = drg_data[drg_group]['los_mean']

    # Convert to DataFrames for analysis
    df = pd.DataFrame(patients)

    # Print summary statistics
    print(f"\n=== Patient Data Generation Summary ===")
    print(f"PTTR Scenario: {pttr_scenario.upper()} ({utilization * 100:.0f}% utilization)")
    print(f"Target Sessions: {target_sessions:,}")
    print(f"Generated Patients: {len(patients):,}")
    print(f"Patient-to-Therapist Ratio: {len(patients) / T:.1f}:1")
    print(f"Total Therapist Capacity: {total_capacity:,} sessions")
    print(f"Average LOS: {avg_mean_unweighted:.2f} sessions")

    print(f"\n=== Horizon Definitions ===")
    print(f"Focus Horizon (D): {min(D)} to {max(D)} ({len(D)} days)")
    print(f"Planning Horizon: {min(D_planning)} to {max(D_planning)} ({len(D_planning)} days)")
    print(f"Extended Horizon (D_ext): {min(D_ext)} to {max(D_ext)} ({len(D_ext)} days)")

    print(f"\n=== DRG Group Distribution ===")
    print(df['drg_group'].value_counts().sort_index())

    print(f"\n=== Estimated Distribution Parameters ===")
    for drg in ['E65A', 'E65B', 'E65C']:
        drg_info = drg_data[drg]
        print(f"\n{drg} ({drg_info['distribution']} distribution):")
        print(f"  Input: min={drg_info['los_min']}, max={drg_info['los_max']}, mean={drg_info['los_mean']}")

        if drg_info['distribution'] == 'lognormal':
            print(f"  Calculated Parameters: μ_L = {drg_info['mu_L']:.3f}, σ_L = {drg_info['sigma_L']:.3f}")
        elif drg_info['distribution'] == 'gamma':
            print(f"  Calculated Parameters: k = {drg_info['k']:.2f}, θ = {drg_info['theta']:.2f}")
            print(f"  Approximated σ = {drg_info['sigma_approx']:.2f}")

    print(f"\n=== Requirements Statistics by DRG Group ===")
    for drg in ['E65A', 'E65B', 'E65C']:
        if drg in df['drg_group'].values:
            drg_info = drg_data[drg]
            print(f"\n{drg} ({drg_info['distribution']} distribution):")
            if drg_info['distribution'] == 'lognormal':
                print(f"  Parameters: μ_L = {drg_info['mu_L']:.3f}, σ_L = {drg_info['sigma_L']:.3f}")
            else:
                print(f"  Parameters: k = {drg_info['k']:.2f}, θ = {drg_info['theta']:.2f}")
            print(df[df['drg_group'] == drg]['requirements'].describe())

    # Generate plots if requested
    if plot_show:
        plt.figure(figsize=(20, 15))

        # DRG distribution
        plt.subplot(3, 2, 1)
        sns.countplot(data=df, x='drg_group', order=['E65A', 'E65B', 'E65C'])
        plt.title('Distribution of DRG Groups')
        plt.xlabel('DRG Group')
        plt.ylabel('Count')

        # Entry day distribution
        plt.subplot(3, 2, 2)
        day_bins = np.arange(min(D_ext), max(D_ext) + 2) - 0.5
        sns.histplot(data=df, x='entry_day', bins=day_bins, discrete=True, stat="density")
        plt.title(f'Density of Entry Days (Range: {min(D_ext)} to {max(D_ext)})')
        plt.xlabel('Day')
        plt.ylabel('Density')

        # Requirements by DRG group
        for idx, drg in enumerate(['E65A', 'E65B', 'E65C']):
            plt.subplot(3, 2, idx + 3)
            drg_data_filtered = df[df['drg_group'] == drg]
            if not drg_data_filtered.empty:
                sns.histplot(data=drg_data_filtered, x='requirements',
                             bins=range(df['requirements'].min(), df['requirements'].max() + 1),
                             discrete=True, stat="density")

                # Add distribution info to title
                dist_info = drg_data[drg]['distribution']
                plt.title(f'Density of Requirements for {drg} ({dist_info})')
            plt.xlabel('Number of Required Treatments')
            plt.ylabel('Density')

        # Overall requirements
        plt.subplot(3, 2, 6)
        sns.histplot(data=df, x='requirements',
                     bins=range(df['requirements'].min(), df['requirements'].max() + 1),
                     discrete=True, stat="density")
        plt.title('Overall Density of Requirements')
        plt.xlabel('Number of Required Treatments')
        plt.ylabel('Density')

        plt.tight_layout()
        plt.show()

    requirements_dict = df.set_index("patient_id")["requirements"].to_dict()
    entry_day_dict = df.set_index("patient_id")["entry_day"].to_dict()

    return requirements_dict, entry_day_dict, Max_t, P, D, D_planning, D_full, list(range(1, T + 1)), M_p, W_coeff