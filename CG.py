import gurobipy as gp
import multiprocessing
import time

from masterproblem import MasterProblem_d
from subproblem import Subproblem
from Utils.compactmodel import Problem_d
from Utils.Generell.instance_setup import *
from Utils.initial_cg_sol import *
from Utils.Generell.plots import *
from Utils.Pre_Patients.pre_patients_heuristic import *


class ColumnGeneration:
    """
    Column Generation class for solving patient scheduling problems.
    Can be used standalone or as part of a Branch-and-Price framework.
    """

    def __init__(self, seed, app_data, T, D_focus, max_itr=100, threshold=1e-5,
                 pttr='medium', show_plots=False, pricing_filtering=True, therapist_agg=False,
                 max_stagnation_itr=5, stagnation_threshold=1e-4, learn_method='pwl', callback_after_iteration=None):
        """
        Initialize Column Generation solver.

        Args:
            seed: Random seed for reproducibility
            app_data: Dictionary with learning parameters
            T: Number of therapists
            D_focus: Number of focus days
            max_itr: Maximum CG iterations
            threshold: Convergence threshold
            pttr: Patient-to-therapist ratio scenario
            show_plots: Whether to show visualization plots
            pricing_filtering: Enable pricing filter for subproblems
            therapist_agg: Enable therapist aggregation
            max_stagnation_itr: Max iterations without dual improvement before termination
            stagnation_threshold: Minimum relative improvement to count as progress
        """
        # Store parameters
        self.seed = seed
        self.app_data = app_data
        self.T_param = T
        self.D_focus = D_focus
        self.max_itr = max_itr
        self.threshold = threshold
        self.pttr = pttr
        self.show_plots = show_plots
        self.pricing_filtering = pricing_filtering
        self.therapist_agg = therapist_agg
        self.max_stagnation_itr = max_stagnation_itr
        self.stagnation_threshold = stagnation_threshold
        self.learn_method = learn_method
        self.callback_after_iteration = callback_after_iteration

        # Initialize random seed
        random.seed(seed)

        # Results storage
        self.iteration_stats = []
        self.master = None
        self.problem = None
        self.global_solutions = None

        # Timing
        self.start_time = None
        self.total_time = None

        # Results
        self.lp_obj = None
        self.frac_info = None
        self.ip_obj = None
        self.com_obj = None
        self.gap = None
        self.is_integral = None
        self.num_iterations = 0

        # Stagnation detection
        self.best_lp_obj = float('inf')
        self.stagnation_counter = 0
        self.lp_obj_history = []

    def setup(self):
        """
        Setup instance: generate data, create initial solution, build models.
        """
        print("=" * 100)
        print(" SETUP PHASE ".center(100, "="))
        print("=" * 100)

        self.start_time = time.time()

        # Extract learning parameters
        self.learning_params = {
            'theta_base': self.app_data['theta_base'][0],
            'lin_increase': self.app_data['lin_increase'][0],
            'k_learn': self.app_data['k_learn'][0],
            'infl_point': self.app_data['infl_point'][0],
            'const': self.app_data['theta_base'][0]
        }

        # Generate patient and therapist data
        print("\n[Setup] Generating patient and therapist data...")
        self.Req, self.Entry, self.Max_t, self.P, self.D, self.D_Ext, self.D_Full, self.T, self.M_p, self.W_coeff = \
            generate_patient_data_log(
                T=self.T_param,
                D_focus=self.D_focus,
                W_on=self.app_data["W_on"][0],
                W_off=self.app_data["W_off"][0],
                daily=self.app_data["daily"][0],
                pttr_scenario=self.pttr,
                seed=self.seed,
                plot_show=self.show_plots
            )

        # Create mappings
        print("[Setup] Creating patient and therapist mappings...")
        self.Nr, self.Req_agg, self.Entry_agg, self.Nr_agg, self.agg_to_patient = \
            get_unique_combinations_and_list_with_dicts(self.Req, self.Entry, self.P)

        self.P_Pre, self.P_F, self.P_Post, self.P_Join, self.E_dict = \
            categorize_patients_full(self.Entry_agg, self.D)

        self.G_C, self.g_j_C, self.Q_jd_Agg, self.therapist_to_type = \
            aggregate_therapists(self.T, self.Max_t, self.app_data["W_on"][0],
                                 self.app_data["W_off"][0], self.D_Ext)
        self.S_Bound = {}
        for p_idx in self.P_Join:
            self.S_Bound[p_idx] = max(10, math.ceil(
                min((self.Req[p_idx] / self.W_coeff) + 2, max(self.D_Ext) - self.Entry[p_idx]) * (
                            self.app_data["MS"][0] - self.app_data["MS_min"][0]) / self.app_data["MS"][0]))

        # Create DataFrame
        print("[Setup] Creating data frame...")
        max_len = max(len(self.P), len(self.P_Pre), len(self.P_Post), len(self.P_F),
                      len(self.P_Join), len(self.T), len(self.D), len(self.D_Full),
                      len(self.D_Ext))

        self.data = pd.DataFrame({
            'P': self.P + [np.nan] * (max_len - len(self.P)),
            'P_Full': self.Nr + [np.nan] * (max_len - len(self.Nr)),
            'P_Pre': self.P_Pre + [np.nan] * (max_len - len(self.P_Pre)),
            'P_Post': self.P_Post + [np.nan] * (max_len - len(self.P_Post)),
            'P_F': self.P_F + [np.nan] * (max_len - len(self.P_F)),
            'G': self.G_C + [np.nan] * (max_len - len(self.G_C)),
            'P_Join': self.P_Join + [np.nan] * (max_len - len(self.P_Join)),
            'T': self.T + [np.nan] * (max_len - len(self.T)),
            'D_Full': self.D_Full + [np.nan] * (max_len - len(self.D_Full)),
            'D_Ext': self.D_Ext + [np.nan] * (max_len - len(self.D_Ext)),
            'D': self.D + [np.nan] * (max_len - len(self.D))
        })

        # Preprocessing
        print("[Setup] Running preprocessing for pre-patients...")
        self.pre_x, self.pre_y, self.pre_los, self.pre_x_filtered, self.pre_x_filt = \
            pre_processing_schedule(
                self.P_Pre, self.P_F, self.T, self.D_Ext, self.Entry_agg, self.Req_agg,
                self.app_data['learn_type'][0], self.learning_params,
                self.app_data['MS'][0], self.app_data['MS_min'][0],
                self.Max_t, self.Nr_agg, self.therapist_to_type
            )
        print('Focus-Patients', self.P_F)
        print('len(Focus-Patients)', len(self.P_F))
        print('Join-Patients', sorted(self.P_F + self.P_Post))
        print('len(Join-Patients)', len(sorted(self.P_F + self.P_Post)))
        # Build compact model
        print("[Setup] Building compact model...")
        self.problem = Problem_d(
            self.data, self.Req, self.Entry, self.Max_t, self.app_data,
            self.pre_x_filt, self.W_coeff, 10, self.learn_method
        )
        self.problem.buildModel()

        # Initialize global solutions storage
        self.global_solutions = {
            'x': {}, 'LOS': {}, 'y': {}, 'z': {}, 'App': {}, 'S': {}, 'l': {}
        }

        # Determine therapist configuration
        if self.therapist_agg:
            self.Max_t_cg = self.Q_jd_Agg
            self.therapist_type = self.therapist_to_type
            patients_for_initial = self.P_F + self.P_Post
            therapist_set = self.G_C
        else:
            self.Max_t_cg = self.Max_t
            self.therapist_type = None
            patients_for_initial = self.P_F + self.P_Post
            therapist_set = self.T

        # Generate initial solution
        print("[Setup] Generating initial CG solution...")
        self.start_x, self.start_los, start_y, start_z, start_App, start_S, start_l, _, _ = \
            initial_cg_starting_sol(
                self.Max_t_cg, patients_for_initial, self.D_Ext, therapist_set,
                self.Req_agg, self.Entry_agg, self.pre_x, self.Nr_agg,
                self.P_F, self.M_p, self.therapist_to_type if self.therapist_agg else None
            )

        self.global_solutions = integrate_initial_solution_to_global(
            self.global_solutions, self.start_x, self.start_los, start_y, start_z,
            start_App, start_S, start_l, self.app_data, 1, self.P_Join
        )

        # Build master problem
        print("[Setup] Building master problem...")
        self.master = MasterProblem_d(
            self.data, self.Max_t_cg, self.Nr_agg, self.Req_agg, self.pre_x, self.E_dict
        )
        self.master.buildModel()
        self.master.startSol(self.start_x, self.start_los)

        print(f"[Setup] Complete! Time: {time.time() - self.start_time:.2f}s")
        print(f"[Setup] Problem size: {len(self.P_Join)} patients, {len(self.T)} therapists, {len(self.D)} days")
        print("=" * 100 + "\n")

    def solve_cg(self):
        """
        Main Column Generation loop with stagnation detection.
        """
        print("=" * 100)
        print(" COLUMN GENERATION ".center(100, "="))
        print("=" * 100 + "\n")

        # Initialize CG loop variables
        next_base_col_idx = 2
        pricing_filter_history = {}
        itr = 0
        skipped_sp = 0
        skipped_sp_post = 0

        # Main CG loop
        while itr < self.max_itr:
            iter_start_time = time.time()
            itr += 1
            base_idx_this_iteration = next_base_col_idx
            max_idx_this_iteration = base_idx_this_iteration - 1

            print(f"\n{'*' * 102}\n*{f'Begin Column Generation Iteration {itr}':^100}*\n{'*' * 102}")

            # Solve master problem
            master_start_time = time.time()
            self.master.solRelModel()
            current_lp_obj = self.master.Model.ObjVal
            duals_pi, duals_gamma = self.master.getDuals()
            master_time = time.time() - master_start_time

            # Store LP objective history
            self.lp_obj_history.append(current_lp_obj)

            # Check for dual improvement (stagnation detection)
            if itr == 1:
                # First iteration - initialize best objective
                self.best_lp_obj = current_lp_obj
                improvement = 0.0
                self.stagnation_counter = 0
                print(f"[Dual] Initial LP objective: {current_lp_obj:.6f}")
            else:
                # Calculate relative improvement (for minimization)
                if abs(self.best_lp_obj) > 1e-10:
                    relative_improvement = (self.best_lp_obj - current_lp_obj) / abs(self.best_lp_obj)
                else:
                    relative_improvement = abs(self.best_lp_obj - current_lp_obj)

                improvement = self.best_lp_obj - current_lp_obj

                # Check if there's significant improvement
                if relative_improvement > self.stagnation_threshold:
                    # Good improvement - reset counter
                    self.stagnation_counter = 0
                    self.best_lp_obj = current_lp_obj
                    print(
                        f"[Dual] LP objective improved: {current_lp_obj:.6f} (Δ={improvement:.6f}, rel={relative_improvement:.4%})")
                else:
                    # No significant improvement - increment counter
                    self.stagnation_counter += 1
                    print(
                        f"[Dual] No significant improvement: {current_lp_obj:.6f} (Δ={improvement:.6f}, rel={relative_improvement:.4%})")
                    print(f"[Dual] Stagnation counter: {self.stagnation_counter}/{self.max_stagnation_itr}")

            # Check stagnation termination criterion
            if self.stagnation_counter >= self.max_stagnation_itr:
                print(f"\n{'*' * 102}")
                print(
                    f"*{f'STAGNATION DETECTED: No dual improvement for {self.max_stagnation_itr} consecutive iterations':^100}*")
                print(f"*{f'Best LP objective: {self.best_lp_obj:.6f}':^100}*")
                print(f"{'*' * 102}\n")
                break

            # Pricing filter: determine which subproblems to solve
            patients_to_solve = self._apply_pricing_filter(
                itr, pricing_filter_history, duals_pi, duals_gamma, skipped_sp, skipped_sp_post
            )

            if not patients_to_solve:
                print("No subproblems to solve after filtering. Terminating.")
                break

            # Solve subproblems in parallel
            print(
                f"Starting subproblem solving for {len(patients_to_solve)} of {len(self.P_Join)} patients on {multiprocessing.cpu_count()} cores.")
            subproblem_start_time = time.time()
            results_from_workers_with_time = self._solve_subproblems_parallel(
                patients_to_solve, duals_gamma, duals_pi
            )
            subproblem_time = time.time() - subproblem_start_time

            # Extract results and worker runtimes
            results_from_workers = [res[0] for res in results_from_workers_with_time]
            worker_runtimes = [res[1] for res in results_from_workers_with_time]

            # Process results and add columns to master
            modelImprovable, new_cols_added_count, best_results_for_history, max_idx_this_iteration = \
                self._process_subproblem_results(
                    results_from_workers, base_idx_this_iteration, max_idx_this_iteration
                )

            # Update pricing filter history
            for index, reduced_cost in best_results_for_history.items():
                current_duals = {
                    'bar_c': reduced_cost,
                    'gamma_n': duals_gamma[index],
                    'pi_td': duals_pi.copy()
                }
                pricing_filter_history[(index, itr)] = current_duals

            # Collect iteration statistics
            iter_end_time = time.time()
            stats_this_iteration = {
                'Iteration': itr,
                'LP Objective': round(current_lp_obj, 6),
                'Improvement': round(improvement if itr > 1 else 0, 6),
                'Stagnation': self.stagnation_counter,
                'Total Time (s)': round(iter_end_time - iter_start_time, 2),
                'Master Time (s)': round(master_time, 2),
                'Subproblems Time (s)': round(subproblem_time, 2),
                'Subproblems Solved': len(patients_to_solve),
                'Subproblems Skipped': len(self.P_Join) - len(patients_to_solve),
                'Columns Added': new_cols_added_count,
                'Min Worker Time (s)': round(min(worker_runtimes), 2) if worker_runtimes else 0,
                'Max Worker Time (s)': round(max(worker_runtimes), 2) if worker_runtimes else 0,
                'Avg Worker Time (s)': round(sum(worker_runtimes) / len(worker_runtimes), 2) if worker_runtimes else 0
            }
            self.iteration_stats.append(stats_this_iteration)

            # Call external callback if provided
            if self.callback_after_iteration:
                self.callback_after_iteration(itr, self)

            # Check convergence
            if modelImprovable:
                print(
                    f"Added {new_cols_added_count} new columns in iteration {itr}. Highest index used: {max_idx_this_iteration}")
                self.master.Model.update()
                next_base_col_idx = max_idx_this_iteration + 1
            else:
                print(f"\n{'*' * 102}\n*{f'No more improvable columns found in iteration {itr}':^100}*\n{'*' * 102}")
                break

        self.num_iterations = itr

        # Print termination reason
        if itr >= self.max_itr:
            print(f"\nColumn Generation terminated: Maximum iterations ({self.max_itr}) reached.")
        elif self.stagnation_counter >= self.max_stagnation_itr:
            print(
                f"\nColumn Generation terminated: Stagnation detected ({self.max_stagnation_itr} iterations without improvement).")
        else:
            print(f"\nColumn Generation finished after {itr} iterations (convergence).")

    def _apply_pricing_filter(self, itr, pricing_filter_history, duals_pi, duals_gamma,
                              skipped_sp, skipped_sp_post):
        """
        Apply pricing filter to determine which subproblems need to be solved.
        """
        patients_to_solve = []

        if self.pricing_filtering and itr > 1:
            for index in self.P_Join:
                skip_subproblem = False
                previous_iterations = [ell for (n, ell) in pricing_filter_history.keys()
                                       if n == index and ell < itr]

                if previous_iterations:
                    ell = max(previous_iterations)
                    hist = pricing_filter_history[(index, ell)]
                    lb = hist['bar_c'] + hist['gamma_n'] - duals_gamma[index]
                    sum_term = sum(min(0, hist['pi_td'].get(key, 0) - duals_pi.get(key, 0))
                                   for key in hist['pi_td'])
                    lb += sum_term

                    if lb >= -self.threshold:
                        skip_subproblem = True
                        skipped_sp += 1
                        if index in self.P_Post:
                            skipped_sp_post += 1

                if not skip_subproblem:
                    patients_to_solve.append(index)
        else:
            patients_to_solve = self.P_Join

        return patients_to_solve

    def _solve_subproblems_parallel(self, patients_to_solve, duals_gamma, duals_pi):
        """
        Solve subproblems in parallel using multiprocessing.
        """
        duals_delta = 0
        node_path = ''  # Root node

        tasks_args = [
            (index, duals_gamma, duals_pi, duals_delta,
             self.data, self.Req_agg, self.Entry_agg,
             self.app_data, self.W_coeff, self.E_dict, self.therapist_type,
             self.P_F, self.S_Bound, self.learn_method, node_path)
            for index in patients_to_solve
        ]

        with multiprocessing.Pool() as pool:
            results = pool.map(solve_subproblem_for_patient, tasks_args)

        return results

    def _process_subproblem_results(self, results_from_workers, base_idx_this_iteration,
                                    max_idx_this_iteration):
        """
        Process subproblem results and add profitable columns to master problem.
        """
        modelImprovable = False
        new_cols_added_count = 0
        best_results_for_history = {}

        results_dict = {cols[0]['index']: cols for cols in results_from_workers if cols}

        for index in self.P_Join:
            if index in results_dict:
                patient_columns = results_dict[index]
                best_col = min(patient_columns, key=lambda c: c['reduced_cost'])
                best_results_for_history[index] = best_col['reduced_cost']

                new_cols_for_this_patient = 0
                for col_info in patient_columns:
                    modelImprovable = True
                    current_col_id = base_idx_this_iteration + new_cols_for_this_patient
                    max_idx_this_iteration = max(max_idx_this_iteration, current_col_id)

                    # Add column to master
                    self.master.addSchedule(col_info['schedules_x'])
                    self.master.addLOS(col_info['schedules_los'])

                    los_list_for_master = col_info['los_list']
                    if not col_info['is_focus_patient']:
                        los_list_for_master[0] = 0

                    self.master.addLambdaVar(
                        col_info['index'], current_col_id,
                        [*col_info['lambda_list'], *col_info['x_list']],
                        los_list_for_master
                    )

                    # Store solution variables
                    solution_key = (col_info['index'], current_col_id)
                    for var_name, var_value in col_info['solution_vars'].items():
                        self.global_solutions[var_name][solution_key] = var_value

                    new_cols_for_this_patient += 1
                    new_cols_added_count += 1

        return modelImprovable, new_cols_added_count, best_results_for_history, max_idx_this_iteration

    def finalize(self):
        """
        Finalize the solution: solve LP relaxation, check integrality, solve IP.
        """
        print("\n" + "=" * 100)
        print(" FINALIZATION ".center(100, "="))
        print("=" * 100)

        # Solve LP relaxation and check integrality
        print("\n[Finalize] Solving LP relaxation and checking integrality...")
        self.master.solRelModel()
        self.is_integral, self.lp_obj, self.frac_info = self.master.check_fractionality()

        # Solve IP
        print("[Finalize] Solving integer program...")
        self.is_integral, self.lp_obj, self.frac_info = self.master.finSol()
        self.master.Model.write('Final_root.lp')
        self.ip_obj = self.master.Model.objVal

        # Calculate gap
        if self.ip_obj > 0:
            self.gap = abs((self.ip_obj - self.lp_obj) / self.ip_obj)
        else:
            self.gap = 0.0

        # Solve compact model for comparison
        print("[Finalize] Solving compact model...")
        self.problem.solveModel()
        self.comp_obj = self.problem.Model.objVal

        self.total_time = time.time() - self.start_time

    def print_statistics(self):
        """
        Print detailed statistics about the solution process.
        """
        print("\n" + "=" * 100)
        print(" PARALLELIZATION STATISTICS ".center(100, "="))
        print("=" * 100)

        if not self.iteration_stats:
            print("No iterations were completed.")
        else:
            stats_df = pd.DataFrame(self.iteration_stats)
            print(stats_df.to_string(index=False))

            print("\n" + "-" * 100)
            print(" SUMMARY ".center(100, "-"))
            print("-" * 100)

            total_time_cg = stats_df['Total Time (s)'].sum()
            total_master_time = stats_df['Master Time (s)'].sum()
            total_subproblem_time = stats_df['Subproblems Time (s)'].sum()

            print(f"Total time in CG loop: {total_time_cg:.2f} s")
            if total_time_cg > 0:
                print(
                    f"  - Time in Master Problem: {total_master_time:.2f} s ({total_master_time / total_time_cg:.1%})")
                print(
                    f"  - Time in Subproblems:    {total_subproblem_time:.2f} s ({total_subproblem_time / total_time_cg:.1%})")
                other_time = total_time_cg - total_master_time - total_subproblem_time
                print(f"  - Time in Overhead/Other: {other_time:.2f} s ({other_time / total_time_cg:.1%})")

    def print_results(self):
        """
        Print final results.
        """
        print("\n" + "=" * 100)
        print(" FINAL RESULTS ".center(100, "="))
        print("=" * 100)

        print(f"Total runtime: {self.total_time:.2f} seconds")
        print(f"CG completed after {self.num_iterations} iterations.")
        print(f"LP relaxation value: {self.lp_obj:.5f}")
        print(f"Final IP value: {self.ip_obj:.5f}")
        print(f"Final MIP Gap: {self.gap:.5f}")
        print(f"Is integral? {self.is_integral}")
        print(f"Compact model objective: {self.problem.Model.objVal:.5f}")

        print("\n[Results] Active lambda variables:")
        self.master.printLambda()

        print("\n[Results] Length of stay (LOS) for focus patients:")
        self.problem.printLOS()

        print("\n" + "=" * 100 + "\n")

    def export_models(self, master_filename='Master_2.lp', compact_filename='Compact.lp'):
        """
        Export master and compact models to LP files.
        """
        self.master.Model.write(master_filename)
        self.problem.Model.write(compact_filename)
        print(f"[Export] Models saved to {master_filename} and {compact_filename}")

    def solve(self):
        """
        Complete solve: setup, solve CG, finalize, and print results.
        """
        self.setup()
        self.solve_cg()
        self.finalize()
        self.print_statistics()
        self.print_results()

        return {
            'lp_obj': self.lp_obj,
            'ip_obj': self.ip_obj,
            'comp_obj': self.comp_obj,
            'gap': self.gap,
            'is_integral': self.is_integral,
            'num_iterations': self.num_iterations,
            'total_time': self.total_time,
            'compact_obj': self.problem.Model.objVal
        }


def solve_subproblem_for_patient(args):
    """
    Worker function: Solves the subproblem for a patient and returns up to 10
    unique columns with negative reduced costs as well as its own runtime.
    """
    worker_start_time = time.time()

    # Unpack arguments
    (index, duals_gamma, duals_pi, duals_delta, data, Req_agg, Entry_agg, app_data, W_coeff,
     E_dict, therapist_type, P_F, S_Bound, learn_meth, node_path) = args
    max_cols_per_iter = 10

    # Create and solve subproblem
    subproblem = Subproblem(
        data, duals_gamma, duals_pi, duals_delta, index, 0, Req_agg, Entry_agg,
        app_data, W_coeff, E_dict, S_Bound, num_tangents=10, reduction=True, learn_method=learn_meth,  node_path=''
    )
    subproblem.buildModel()

    # Gurobi parameters for solution pool
    subproblem.Model.Params.PoolSearchMode = 2
    subproblem.Model.Params.PoolSolutions = 50
    subproblem.Model.Params.OutputFlag = 0
    subproblem.solModel()

    newly_generated_columns = []

    # Extract unique profitable columns
    if subproblem.Model.SolCount > 0:
        unique_schedules = set()
        threshold = 1e-3

        for i in range(subproblem.Model.SolCount):
            if len(newly_generated_columns) >= max_cols_per_iter:
                break

            subproblem.Model.setParam(gp.GRB.Param.SolutionNumber, i)
            reducedCost = subproblem.Model.PoolObjVal

            if reducedCost < -threshold:
                schedules_x, x_list, _ = subproblem.getOptVals('x')
                schedules_x_tuple = tuple(sorted(schedules_x.items()))

                if schedules_x_tuple not in unique_schedules:
                    unique_schedules.add(schedules_x_tuple)
                    schedules_los, los_list, _ = subproblem.getOptVals('LOS')

                    solution_vars = {
                        var: subproblem.getVarSol(var, 0)
                        for var in ['x', 'LOS', 'y', 'z', 'S', 'l']
                    }

                    if app_data["learn_type"][0] in ['exp', 'sigmoid', 'lin']:
                        solution_vars['App'] = subproblem.getVarSol('App', 0)

                    column_info = {
                        'index': index,
                        'is_focus_patient': index in P_F,
                        'reduced_cost': reducedCost,
                        'schedules_x': schedules_x,
                        'x_list': x_list,
                        'schedules_los': schedules_los,
                        'los_list': los_list,
                        'lambda_list': subproblem.create_lambda_list(index),
                        'solution_vars': solution_vars
                    }
                    newly_generated_columns.append(column_info)

    worker_end_time = time.time()
    return (newly_generated_columns, worker_end_time - worker_start_time)