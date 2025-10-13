from CG import ColumnGeneration
from branch_and_price import BranchAndPrice
from Utils.compact import Problem_d
import pandas as pd


def run_single_instance(seed):
    """
    Führt eine einzelne Instanz des Algorithmus mit einem gegebenen Seed aus.
    """
    # ===========================
    # CONFIGURATION PARAMETERS
    # ===========================
    app_data = {
        'learn_type': ['lin'],
        'theta_base': [0.02],
        'lin_increase': [0.01],
        'k_learn': [0.01],
        'infl_point': [2],
        'MS': [5],
        'MS_min': [2],
        'W_on': [6],
        'W_off': [1],
        'daily': [4]
    }
    T = 3
    D_focus = 5
    dual_improvement_iter = 5
    dual_stagnation_threshold = 1e-4
    max_itr = 100
    threshold = 1e-3
    pttr = 'medium'
    show_plots = False
    pricing_filtering = True
    therapist_agg = False
    learn_method = 'pwl'
    use_branch_and_price = True
    branching_strategy = 'mp'
    solve_and_compare_compact_model = True

    # ===========================
    # SETUP CG SOLVER
    # ===========================
    print(f"\n{'=' * 100}\n" + f" RUNNING FOR SEED: {seed} ".center(100, "=") + f"\n{'=' * 100}")

    cg_solver = ColumnGeneration(
        seed=seed,
        app_data=app_data,
        T=T,
        D_focus=D_focus,
        max_itr=max_itr,
        threshold=threshold,
        pttr=pttr,
        show_plots=show_plots,
        pricing_filtering=pricing_filtering,
        therapist_agg=therapist_agg,
        max_stagnation_itr=dual_improvement_iter,
        stagnation_threshold=dual_stagnation_threshold,
        learn_method=learn_method
    )
    cg_solver.setup()

    # ===========================
    # SOLVE
    # ===========================
    if use_branch_and_price:
        bnp_solver = BranchAndPrice(cg_solver, branching_strategy=branching_strategy, verbose=False,
                                    ip_heuristic_frequency=0, early_incumbent_iteration=2)
        results = bnp_solver.solve(time_limit=3600, max_nodes=100)
    else:
        results = cg_solver.solve()

    # ===========================
    # SOLVE COMPACT MODEL FOR COMPARISON
    # ===========================
    if solve_and_compare_compact_model:
        cg_solver.problem.solveModel()
        compact_obj_val = cg_solver.problem.Model.objVal
        results['compact_obj'] = compact_obj_val

    return results


def main():
    """
    Main function to run the algorithm over multiple seeds and compare results.
    """
    num_runs = 100  # Anzahl der Seeds, die getestet werden sollen (1 bis 100)
    all_results = []

    for seed in range(1, num_runs + 1):
        try:
            result = run_single_instance(seed)
            result['seed'] = seed
            all_results.append(result)
        except Exception as e:
            print(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(f" FEHLER BEI SEED {seed}: {e} ".center(100, "X"))
            print(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            all_results.append({'seed': seed, 'error': str(e)})

    # ===========================
    # FINAL SUMMARY OF ALL RUNS
    # ===========================
    print("\n" + "=" * 100)
    print(" FINAL SUMMARY OF ALL SEED RUNS ".center(100, "="))
    print("=" * 100)

    if not all_results:
        print("No runs were successfully completed.")
        return

    # Erstelle ein DataFrame zur einfachen Analyse
    df = pd.DataFrame(all_results)
    df['final_obj'] = df.apply(lambda row: row.get('incumbent') if row.get('use_branch_and_price', True) else row.get('ip_obj'), axis=1)

    successful_runs = df[df['error'].isnull()]
    failed_runs = df[df['error'].notnull()]

    print(f"Total runs attempted: {num_runs}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Failed runs: {len(failed_runs)}")

    if not successful_runs.empty:
        print("\nComparison of Objective Values (B&P/CG vs. Compact Model):")

        # Vergleiche die Zielfunktionswerte
        successful_runs['objectives_match'] = abs(successful_runs['final_obj'] - successful_runs['compact_obj']) < 1e-4

        matching_count = successful_runs['objectives_match'].sum()
        mismatching_count = len(successful_runs) - matching_count

        print(f"  - Runs with IDENTICAL objective values: {matching_count}")
        print(f"  - Runs with DIFFERENT objective values: {mismatching_count}")

        if mismatching_count > 0:
            print("\n  Seeds with differing objective values:")
            mismatched_seeds = successful_runs[~successful_runs['objectives_match']]
            for _, row in mismatched_seeds.iterrows():
                print(f"    - Seed {row['seed']}: B&P/CG Obj = {row['final_obj']:.5f}, Compact Obj = {row['compact_obj']:.5f}")
        else:
            print("\n  ✅ Alle erfolgreichen Durchläufe hatten identische Zielfunktionswerte zwischen dem B&P/CG-Ansatz und dem kompakten Modell.")

    if not failed_runs.empty:
        print("\nDetails of failed runs:")
        for _, row in failed_runs.iterrows():
            print(f"  - Seed {row['seed']}: Error: {row['error']}")

    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
