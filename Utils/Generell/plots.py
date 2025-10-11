import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_app_efficiency(flat_list, D, P):
    if len(flat_list) % D != 0:
        raise ValueError("The length of the list must be a multiple of D.")

    # Split data for each patient
    patient_data = {}
    for idx, p_id in enumerate(P):
        start_idx = idx * D
        end_idx = (idx + 1) * D
        patient_data[p_id] = flat_list[start_idx:end_idx]

    # X-axis: days from 1 to D
    days = list(range(1, D + 1))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate global y_min and y_max across all patients
    all_values = np.concatenate([np.array(values) for values in patient_data.values()])
    y_min = np.nanmin(all_values)
    y_max = np.nanmax(all_values)
    epsilon = (y_max-y_min)/30

    for patient, values in patient_data.items():
        values = np.array(values)
        line, = ax.plot(days, values, label=f'Patient {patient}', marker='o')

        # Find the last valid (non-NaN) value
        valid_indices = np.where(~np.isnan(values))[0]
        if len(valid_indices) > 0:
            last_valid_index = valid_indices[-1]
            x_val = days[last_valid_index]

        ax.vlines(x_val, y_min - epsilon, y_max + epsilon, linestyle='--', color=line.get_color(), alpha = 0.3)

    # Axis labels and title
    ax.set_xlabel('Days (D)')
    ax.set_ylabel('Value')
    ax.set_title('Values per patient over days')

    # Remove grid
    ax.grid(False)

    # Horizontal legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
             fancybox=True, shadow=False, ncol=len(P))

    plt.tight_layout()
    plt.show()

def plot_statistics(file_path, y_column, x_label=r'App Efficiency $\theta$', y_label=r'$\sum LOS$'):
    """
    Create a plot of mean values with whiskers (min/max) for each pttr/theta_base combination.

    Parameters:
    - file_path (str): Path to the Excel file containing the data.
    - y_column (str): Column name for the y-axis data (e.g., 'obj_1_ip').
    - x_label (str): Label for the x-axis (default: LaTeX formatted 'App Efficiency').
    - y_label (str): Label for the y-axis (default: LaTeX formatted '∑ LOS').

    Returns:
    - None: Displays the plot.
    """
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Keep only feasible entries (infeasible? == False)
    df_feasible = df[df['infeasible?'] == False]

    # Get unique pttr values
    pttr_values = df_feasible['pttr'].unique()

    # Assign colors for different pttr values
    colors = plt.cm.tab10(np.linspace(0, 1, len(pttr_values)))
    color_dict = dict(zip(pttr_values, colors))

    # Get unique theta_base values
    baselevel_values = sorted(df_feasible['theta_base'].unique())

    # Prepare data for plotting
    stats = []
    for pttr in pttr_values:
        for theta_base in baselevel_values:
            # Filter data for current pttr/theta_base combination
            data = df_feasible[(df_feasible['pttr'] == pttr) & (df_feasible['theta_base'] == theta_base)][y_column]
            if not data.empty:
                mean_val = data.mean()
                min_val = data.min()
                max_val = data.max()
                std_val = data.std()
                stats.append({
                    'pttr': pttr,
                    'theta_base': theta_base,
                    'mean': mean_val,
                    'min': min_val,
                    'max': max_val,
                    'std': std_val
                })

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats)

    # Create the plot
    plt.figure(figsize=(10, 6))

    for pttr in pttr_values:
        pttr_data = stats_df[stats_df['pttr'] == pttr]
        if not pttr_data.empty:
            # Plot mean values with customized legend label
            plt.plot(pttr_data['theta_base'], pttr_data['mean'], marker='o', label=f'PTTR {pttr.capitalize()}', color=color_dict[pttr])
            # Add whiskers (min/max)
            plt.errorbar(pttr_data['theta_base'], pttr_data['mean'],
                         yerr=[pttr_data['mean'] - pttr_data['min'], pttr_data['max'] - pttr_data['mean']],
                         fmt='none', ecolor=color_dict[pttr], capsize=5, alpha=0.5)

    # Format the plot
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.show()

def plot_3d_statistics(file_path, y1_column, y2_column,
                       x_label=r'App Efficiency $\theta$',
                       y1_label=r'$\sum_{i\in \mathcal{I}} LOS_i$',
                       y2_label=r'$\sum_{i\in \mathcal{I}} d^+_i$'):
    """
    Create a 3D plot of mean values with whiskers (min/max) for each pttr/theta_base combination.

    Parameters:
    - file_path (str): Path to the Excel file containing the data.
    - y1_column (str): Column name for the first y-axis data (e.g., 'obj_lp1').
    - y2_column (str): Column name for the second y-axis data (e.g., 'obj_lp2').
    - x_label (str): Label for the x-axis (default: LaTeX formatted 'App Efficiency').
    - y1_label (str): Label for the first y-axis (default: LaTeX formatted '∑ LOS₁').
    - y2_label (str): Label for the second y-axis (default: LaTeX formatted '∑ LOS₂').

    Returns:
    - None: Displays the plot.
    """
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Keep only feasible entries (infeasible? == False)
    df_feasible = df[df['infeasible?'] == False]

    # Get unique pttr values
    pttr_values = df_feasible['pttr'].unique()

    # Assign colors for different pttr values
    colors = plt.cm.tab10(np.linspace(0, 1, len(pttr_values)))
    color_dict = dict(zip(pttr_values, colors))

    # Get unique theta_base values
    baselevel_values = sorted(df_feasible['theta_base'].unique())

    # Prepare data for plotting
    stats = []
    for pttr in pttr_values:
        for theta_base in baselevel_values:
            # Filter data for current pttr/theta_base combination
            data_y1 = df_feasible[(df_feasible['pttr'] == pttr) & (df_feasible['theta_base'] == theta_base)][y1_column]
            data_y2 = df_feasible[(df_feasible['pttr'] == pttr) & (df_feasible['theta_base'] == theta_base)][y2_column]

            if not data_y1.empty and not data_y2.empty:
                stats.append({
                    'pttr': pttr,
                    'theta_base': theta_base,
                    'mean_y1': data_y1.mean(),
                    'min_y1': data_y1.min(),
                    'max_y1': data_y1.max(),
                    'mean_y2': data_y2.mean(),
                    'min_y2': data_y2.min(),
                    'max_y2': data_y2.max()
                })

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for pttr in pttr_values:
        pttr_data = stats_df[stats_df['pttr'] == pttr]
        if not pttr_data.empty:
            # Plot mean values as points
            ax.scatter(pttr_data['theta_base'], pttr_data['mean_y1'], pttr_data['mean_y2'],
                       label=f'PTTR {pttr.capitalize()}', color=color_dict[pttr], s=50)

            # Add whiskers for y1 (obj_lp1)
            for i, row in pttr_data.iterrows():
                # Vertical whiskers for y1
                ax.plot([row['theta_base'], row['theta_base']],
                        [row['min_y1'], row['max_y1']],
                        [row['mean_y2'], row['mean_y2']],
                        color=color_dict[pttr], alpha=0.5, linewidth=1)

                # Whiskers for y2 (obj_lp2)
                ax.plot([row['theta_base'], row['theta_base']],
                        [row['mean_y1'], row['mean_y1']],
                        [row['min_y2'], row['max_y2']],
                        color=color_dict[pttr], alpha=0.5, linewidth=1)

                # Add caps to whiskers
                # Caps for y1 whiskers
                ax.plot([row['theta_base']], [row['min_y1']], [row['mean_y2']],
                        '_', color=color_dict[pttr], markersize=8, alpha=0.7)
                ax.plot([row['theta_base']], [row['max_y1']], [row['mean_y2']],
                        '_', color=color_dict[pttr], markersize=8, alpha=0.7)

                # Caps for y2 whiskers
                ax.plot([row['theta_base']], [row['mean_y1']], [row['min_y2']],
                        '_', color=color_dict[pttr], markersize=8, alpha=0.7)
                ax.plot([row['theta_base']], [row['mean_y1']], [row['max_y2']],
                        '_', color=color_dict[pttr], markersize=8, alpha=0.7)

    # Format the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y1_label)
    ax.set_zlabel(y2_label)
    ax.legend()
    ax.grid(True)

    # Improve viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()