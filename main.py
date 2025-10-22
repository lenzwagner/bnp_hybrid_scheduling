from CG import ColumnGeneration
from branch_and_price import BranchAndPrice
from logging_config import setup_logging, get_logger

logger = get_logger(__name__)

def main():
    """
    Main function to run Column Generation or Branch-and-Price algorithm.
    """
    # ===========================
    # LOGGING CONFIGURATION
    # ===========================
    setup_logging(log_level='INFO', log_to_file=True, log_dir='logs/info')
    setup_logging(log_level='DEBUG', log_to_file=True, log_dir='logs')
    setup_logging(log_level='ERROR', log_to_file=True, log_dir='logs')
    setup_logging(log_level='WARNING', log_to_file=True, log_dir='logs')


    logger.info("=" * 100)
    logger.info("STARTING BRANCH-AND-PRICE SOLVER")
    logger.info("=" * 100)

    # ===========================
    # CONFIGURATION PARAMETERS
    # ===========================

    # Random seed
    seed = 92

    # Learning parameters
    app_data = {
        'learn_type': ['lin'],  # Learning curve type: 'exp', 'sigmoid', 'lin', or numeric value
        'theta_base': [0.02],  # Base effectiveness
        'lin_increase': [0.01],  # Linear increase rate (for 'lin' type)
        'k_learn': [0.01],  # Learning rate (for 'exp' and 'sigmoid')
        'infl_point': [2],  # Inflection point (for 'sigmoid')
        'MS': [5],  # Maximum session window
        'MS_min': [2],  # Minimum sessions in window
        'W_on': [6],  # Work days per week
        'W_off': [1],  # Days off per week
        'daily': [4]  # Daily capacity per therapist
    }

    # Instance parameters
    T = 3  # Number of therapists
    D_focus = 5  # Number of focus days

    # Algorithm parameters
    dual_improvement_iter = 20  # Max Iterations without dual improvement
    dual_stagnation_threshold = 1e-5
    max_itr = 100  # Maximum CG iterations
    threshold = 1e-5  # Convergence threshold

    # Additional settings
    pttr = 'medium'  # Patient-to-therapist ratio: 'low', 'medium', 'high'
    show_plots = False  # Show visualization plots
    pricing_filtering = True  # Enable pricing filter
    therapist_agg = False  # Enable therapist aggregation
    learn_method = 'pwl'

    # Logger info
    logger.info(f"Configuration: seed={seed}, T={T}, D_focus={D_focus}, pttr={pttr}")

    # Branch-and-Price settings
    use_branch_and_price = True  # Set to False for standard CG
    branching_strategy = 'sp'  # 'mp' for MP variable branching, 'sp' for SP variable branching
    search_strategy = 'dfs' # 'dfs' for Depth-First, 'bfs' for Best-Fit-Search


    # Visualization settings
    visualize_tree = False  # Enable tree visualization
    tree_layout = 'hierarchical'  # 'hierarchical' or 'radial'
    detailed_tree = False  # Show detailed info on nodes
    save_tree_path = 'bnp_tree.png'  # Path to save (None to not save)

    # ===========================
    # CONFIGURATION SUMMARY
    # ===========================

    print("\n" + "=" * 100)
    print(" STARTING SETUP ".center(100, "="))
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  - Mode: {'Branch-and-Price' if use_branch_and_price else 'Column Generation'}")
    if use_branch_and_price:
        print(f"  - Branching Strategy: {branching_strategy.upper()}")
        print(f"  - Search Strategy: {'Depth-First (DFS)' if search_strategy == 'dfs' else 'Best-Fit (BFS)'}")
    print(f"  - Seed: {seed}")
    print(f"  - Learning type: {app_data['learn_type'][0]}")
    print(f"  - Learning method: {learn_method}")
    print(f"  - Therapists: {T}")
    print(f"  - Focus days: {D_focus}")
    print(f"  - Max CG iterations: {max_itr}")
    print(f"  - Threshold: {threshold}")
    print(f"  - PTTR scenario: {pttr}")
    print(f"  - Pricing filtering: {pricing_filtering}")
    print()

    # ===========================
    # SETUP CG SOLVER
    # ===========================

    # Create CG solver
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

    # Setup
    cg_solver.setup()

    # ===========================
    # SOLVE
    # ===========================

    if use_branch_and_price:
        # Branch-and-Price
        print("\n" + "=" * 100)
        print(" INITIALIZING BRANCH-AND-PRICE ".center(100, "="))
        print("=" * 100 + "\n")

        bnp_solver = BranchAndPrice(cg_solver,
                                    branching_strategy=branching_strategy,
                                    search_strategy=search_strategy,
                                    verbose=True,
                                    ip_heuristic_frequency=2,
                                    early_incumbent_iteration=0)
        results = bnp_solver.solve(time_limit=3600, max_nodes=1000)

        # Extract optimal schedules
        if results['incumbent'] is not None:
            print("\n" + "=" * 100)
            print(" EXTRACTING OPTIMAL SCHEDULES ".center(100, "="))
            print("=" * 100)

            optimal_schedules = bnp_solver.extract_optimal_schedules()

            # Print example schedules
            if optimal_schedules:
                p_focus_patients = {
                    patient_id: info
                    for patient_id, info in optimal_schedules['patient_schedules'].items()
                    if info['profile'] in cg_solver.P_F
                }

                # Print first 3 patient schedules as examples
                patient_ids = list(p_focus_patients.keys())[:3]
                for patient_id in patient_ids:
                    bnp_solver.print_detailed_schedule(
                        patient_id,
                        p_focus_patients[patient_id]
                    )

            # Export to CSV
            bnp_solver.export_schedules_to_csv('results/optimal_schedules.csv')

            print("\n" + "=" * 100)
            print(" SCHEDULE EXTRACTION COMPLETE ".center(100, "="))
            print("=" * 100)

        # Print CG statistics (from root node)
        print("\n" + "=" * 100)
        print(" COLUMN GENERATION STATISTICS (ROOT NODE) ".center(100, "="))
        print("=" * 100 + "\n")
        cg_solver.print_statistics()

        # Visualize tree
        if visualize_tree:
            print("\n" + "=" * 100)
            print(" GENERATING TREE VISUALIZATION ".center(100, "="))
            print("=" * 100 + "\n")

            import os
            os.makedirs("Pictures/Tree", exist_ok=True)
            bnp_solver.visualize_tree(layout='hierarchical', save_path='Pictures/Tree/tree_hierarchical.png')
            bnp_solver.visualize_tree(layout='radial', save_path='Pictures/Tree/tree_radial.png')
            bnp_solver.visualize_tree(detailed=True, save_path='Pictures/Tree/tree_detailed.png')

    else:
        # Standard Column Generation
        results = cg_solver.solve()

    # ===========================
    # SUMMARY
    # ===========================

    print("\n" + "=" * 100)
    print(" EXECUTION SUMMARY ".center(100, "="))
    print("=" * 100)
    print(f"Completed successfully!")
    print(f"  - Mode: {'Branch-and-Price' if use_branch_and_price else 'Column Generation'}")
    print(f"  - Total time: {results['total_time']:.2f}s")

    if use_branch_and_price:
        print(f"\nBranch-and-Price Results:")
        print(f"  - Branching strategy: {branching_strategy.upper()}")
        print(f"  - Search strategy: {'Depth-First (DFS)' if search_strategy == 'dfs' else 'Best-Fit (BFS)'}")
        print(f"  - Nodes explored: {results['nodes_explored']}")
        print(f"  - Nodes fathomed: {results['nodes_fathomed']}")
        print(f"  - Nodes branched: {results.get('nodes_branched', 0)}")
        print(f"  - LP bound (LB): {results['lp_bound']:.5f}")
        if results['incumbent']:
            print(f"  - Incumbent (UB): {results['incumbent']:.5f}")
            print(f"  - Gap: {results['gap']:.5%}")
        else:
            print(f"  - Incumbent (UB): None")
        print(f"  - Integral: {results['is_integral']}")
        print(f"  - CG iterations (root): {results['cg_iterations']}")
        print(f"  - IP solves: {results['ip_solves']}")
    else:
        print(f"\nColumn Generation Results:")
        print(f"  - Iterations: {results['num_iterations']}")
        print(f"  - LP objective: {results['lp_obj']:.5f}")
        print(f"  - IP objective: {results['ip_obj']:.5f}")
        print(f"  - Compact model: {results['comp_obj']:.5f}")
        print(f"  - Gap: {results['gap']:.5%}")
        print(f"  - Integral?: {results['is_integral']}")

    print("=" * 100 + "\n")
    return results

if __name__ == "__main__":
    results = main()