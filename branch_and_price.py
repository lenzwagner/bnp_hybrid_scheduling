import sys
import time
from bnp_node import BnPNode
import gurobipy as gu
import copy
import logging


class BranchAndPrice:
    """
    Branch-and-Price Algorithm

    Attributes:
        nodes: Dictionary of all nodes {node_id -> BnPNode}
        node_counter: Counter for unique node IDs
        open_nodes: List of open nodes (later: queue for DFS)
        incumbent: Best found IP solution (upper bound)
        best_lp_bound: Best LP bound of all nodes (lower bound)
        gap: Optimality gap
        cg_solver: Reference to Column Generation solver
    """

    def __init__(self, cg_solver, branching_strategy='mp', search_strategy='dfs', verbose=True,
                 ip_heuristic_frequency=10, early_incumbent_iteration=0):
        """
        Initialize Branch-and-Price with existing CG solver.

        Args:
            cg_solver: ColumnGeneration object (already initialized with setup())
            branching_strategy: 'mp' for MP variable branching, 'sp' for SP variable branching
            search_strategy: 'dfs' for Depth-First-Search or 'bfs' for Best-Fit-Search.
            verbose: If True, print detailed progress
            ip_heuristic_frequency: Solve RMP as IP every N nodes (0 to disable)
            early_incumbent_iteration: CG iteration to compute initial incumbent
                                      - If 0 or None: solve final RMP as IP (after CG converges)
                                      - If > 0: solve RMP as IP after this iteration,
                                               then continue CG without further IP solves
        """
        # Logger
        self.logger = logging.getLogger(__name__)

        # Node management
        self.nodes = {}  # {node_id -> BnPNode}
        self.node_counter = 0
        self.open_nodes = []  # For DFS: list of IDs. For BFS: list of (bound, ID)

        # Output control
        self.verbose = verbose

        # Global bounds
        self.incumbent = float('inf')  # Best IP solution (upper bound)
        self.incumbent_solution = None
        self.incumbent_lambdas = None
        self.best_lp_bound = float('inf')  # Best LP bound (lower bound)
        self.gap = float('inf')

        # Reference to CG solver
        self.cg_solver = cg_solver

        # Search and Branching Configuration
        self.branching_strategy = branching_strategy
        self.search_strategy = search_strategy

        # IP Heuristic
        self.ip_heuristic_frequency = ip_heuristic_frequency

        # Early incumbent computation
        self.early_incumbent_iteration = early_incumbent_iteration if early_incumbent_iteration else 0
        self.incumbent_computed_early = False

        # Start solutions
        self.start_x = self.cg_solver.start_x
        self.start_los = self.cg_solver.start_los

        # Statistics
        self.stats = {
            'nodes_explored': 0,
            'nodes_fathomed': 0,
            'nodes_branched': 0,
            'total_cg_iterations': 0,
            'total_time': 0,
            'incumbent_updates': 0,
            'ip_solves': 0,
            'node_processing_order': [],
            'bfs_decision_log': []
        }

        # Timing
        self.start_time = None

        self.logger.info("\n" + "=" * 100)
        self.logger.info(" BRANCH-AND-PRICE INITIALIZED ".center(100, "="))
        self.logger.info("=" * 100)
        self.logger.info(f"CG Solver ready with {len(self.cg_solver.P_Join)} patients")
        self.logger.info(f"Branching strategy: {self.branching_strategy.upper()}")
        self.logger.info(f"Search strategy: {'Depth-First (DFS)' if self.search_strategy == 'dfs' else 'Best-Fit (BFS)'}")
        if self.early_incumbent_iteration > 0:
            self.logger.info(f"Incumbent strategy: Compute after CG iteration {self.early_incumbent_iteration}")
        else:
            self.logger.info(f"Incumbent strategy: Compute after CG convergence (final RMP as IP)")

        self.logger.info("=" * 100 + "\n")

        # Initialize LP-folder
        import os
        os.makedirs("results", exist_ok=True)
        os.makedirs("LPs/MP/LPs", exist_ok=True)
        os.makedirs("LPs/MP/SOLs", exist_ok=True)
        os.makedirs("LPs/SPs/pricing", exist_ok=True)

        # Logger init
        self.logger.info("=" * 100)
        self.logger.info(" BRANCH-AND-PRICE INITIALIZED ".center(100, "="))
        self.logger.info("=" * 100)
        self.logger.info(f"CG Solver ready with {len(self.cg_solver.P_Join)} patients")
        self.logger.info(f"Branching strategy: {self.branching_strategy.upper()}")

    def _print(self, *args, **kwargs):
        """Print only if verbose mode is enabled."""
        if self.verbose:
            print(*args, **kwargs)

    def _print_always(self, *args, **kwargs):
        """Always print (for critical messages and final results)."""
        print(*args, **kwargs)

    def _early_incumbent_callback(self, iteration, cg_solver):
        """
        Callback executed after each CG iteration at root node.

        If early_incumbent_iteration is set and we reach that iteration,
        solve the RMP as IP to get the initial incumbent.

        Args:
            iteration: Current CG iteration number
            cg_solver: Reference to ColumnGeneration instance
        """
        # Check if we should compute early incumbent
        if self.early_incumbent_iteration == 0:
            return  # No early incumbent requested

        if iteration != self.early_incumbent_iteration:
            return  # Not the right iteration yet

        if self.incumbent_computed_early:
            return  # Already computed

        # Compute incumbent at this iteration
        self.logger.info(f"\n{'─' * 100}")
        self.logger.info(f" COMPUTING EARLY INCUMBENT (after CG iteration {iteration}) ".center(100, "─"))
        self.logger.info(f"{'─' * 100}")
        self.logger.info(f"Solving RMP as IP with columns generated so far...")
        self.logger.info(f"CG will continue afterwards until convergence.\n")

        success = self._solve_rmp_as_ip(cg_solver.master, context="Early Incumbent")

        if success:
            self.incumbent_computed_early = True
            self.logger.info(f"\n✅ Early incumbent computed successfully!")
            self.logger.info(f"   Incumbent: {self.incumbent:.6f}")
            self.logger.info(f"   CG will continue to convergence...\n")
        else:
            self.logger.warning(f"\n⚠️  Early incumbent computation unsuccessful")
            self.logger.warning(f"   CG will continue and we'll try again after convergence.\n")

        self.logger.info(f"{'─' * 100}\n")

    def create_root_node(self):
        """
        Create root node with initial columns from CG heuristic.

        Returns:
            BnPNode: The root node (ID=0, depth=0)
        """
        node = BnPNode(node_id=0, depth=0)

        # Transfer initial columns from CG solver
        for (p, old_col_id) in self.cg_solver.global_solutions.get('x', {}).keys():
            col_id = old_col_id

            # Extract schedules_x from global_solutions
            x_solution = self.cg_solver.global_solutions['x'][(p, old_col_id)]

            # Remap keys: (p, agent, period, old_iteration) -> (p, agent, period, col_id)
            schedules_x = {}
            for (p_key, agent, period, old_iter), value in x_solution.items():
                # Use col_id instead of old_iter
                schedules_x[(p_key, agent, period, 0)] = value

            # Extract schedules_los
            schedules_los = {}
            if (p, old_col_id) in self.cg_solver.global_solutions.get('LOS', {}):
                los_solution = self.cg_solver.global_solutions['LOS'][(p, old_col_id)]
                # Remap keys: (p, old_iteration) -> (p, col_id)
                for (p_key, old_iter), value in los_solution.items():
                    schedules_los[(p_key, 0)] = value

            # Create column data with CORRECT field names
            col_data = {
                'index': p,
                'column_id': col_id,
                'schedules_x': schedules_x,
                'schedules_los': schedules_los,
                'x_list': list(schedules_x.values()),
                'los_list': list(schedules_los.values()),
            }

            # Add other solution data
            for var_name in ['y', 'z', 'S', 'l']:
                if (p, old_col_id) in self.cg_solver.global_solutions.get(var_name, {}):
                    col_data[f'{var_name}_data'] = self.cg_solver.global_solutions[var_name][(p, old_col_id)]

            if 'App' in self.cg_solver.global_solutions:
                if (p, old_col_id) in self.cg_solver.global_solutions['App']:
                    col_data['App_data'] = self.cg_solver.global_solutions['App'][(p, old_col_id)]

            # Store in node pool with the correct column_id
            node.column_pool[(p, col_id)] = col_data

        print(node.column_pool[(55, 1)])
        print('ÄÄÄ')

        # Store node
        self.nodes[0] = node
        if self.search_strategy == 'dfs':
            self.open_nodes.append(0)
        else:  # bfs
            # Root node has no initial bound yet, will be computed in solve_root_node
            self.open_nodes.append((float('inf'), 0))

        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" ROOT NODE CREATED ".center(100, "="))
        self.logger.info(f"{'=' * 100}")
        self.logger.info(f"Root node initialized with {len(node.column_pool)} initial columns")
        self.logger.info(f"Columns distribution:")

        # Show column distribution by profiles
        col_per_profile = {}
        for (p, _) in node.column_pool.keys():
            col_per_profile[p] = col_per_profile.get(p, 0) + 1

        for p in sorted(col_per_profile.keys())[:5]:
            self.logger.info(f"  Profile {p}: {col_per_profile[p]} columns")
        if len(col_per_profile) > 5:
            self.logger.info(f"  ... and {len(col_per_profile) - 5} more profiles")

        # Debug: Show sample column structure
        if node.column_pool:
            sample_key = list(node.column_pool.keys())[0]
            sample_col = node.column_pool[sample_key]
            self.logger.info(f"\n  Sample column {sample_key}:")
            self.logger.info(f"    schedules_x: {len(sample_col.get('schedules_x', {}))} entries")
            self.logger.info(f"    schedules_los: {len(sample_col.get('schedules_los', {}))} entries")
            self.logger.info(f"    x_list: {len(sample_col.get('x_list', []))} values")
            self.logger.info(f"    los_list: {len(sample_col.get('los_list', []))} values")

            # Show first key format
            if sample_col.get('schedules_x'):
                first_key = list(sample_col['schedules_x'].keys())[0]
                self.logger.info(f"    First schedules_x key: {first_key}")

        self.logger.info(f"{'=' * 100}\n")

        return node

    def solve_root_node(self):
        """
        Solve root node via Column Generation.

        Depending on early_incumbent_iteration:
        - If 0: Solve RMP as IP after CG converges
        - If > 0: RMP is solved as IP during CG (via callback),
                  then solve final LP after convergence

        Returns:
            tuple: (lp_bound, is_integral, most_frac_info)
        """
        self.logger.info("\n" + "=" * 100)
        self.logger.info(" SOLVING ROOT NODE ".center(100, "="))
        self.logger.info("=" * 100 + "\n")

        # Log processing of root node
        self.stats['node_processing_order'].append(0)

        # Setup callback if early incumbent is requested
        if self.early_incumbent_iteration > 0:
            self.cg_solver.callback_after_iteration = self._early_incumbent_callback
            self.logger.info(
                f"[Root] Early incumbent will be computed after CG iteration {self.early_incumbent_iteration}\n")
        else:
            self.cg_solver.callback_after_iteration = None
            self.logger.info(f"[Root] Incumbent will be computed after CG convergence\n")

        # Solve with Column Generation
        self.cg_solver.solve_cg()

        # After CG converges: Check if we need to compute incumbent
        if not self.incumbent_computed_early:
            self.logger.info("\n" + "=" * 100)
            self.logger.info(" COMPUTING FINAL INCUMBENT ".center(100, "="))
            self.logger.info("=" * 100)
            self.logger.info("Column Generation converged. All columns generated.")
            self.logger.info("Solving final Root Master Problem as IP to get initial upper bound...\n")

            self._compute_final_incumbent()
        else:
            self.logger.info("\n" + "=" * 100)
            self.logger.info(" USING EARLY INCUMBENT ".center(100, "="))
            self.logger.info("=" * 100)
            self.logger.info(f"Incumbent was already computed at iteration {self.early_incumbent_iteration}")
            self.logger.info(f"Current incumbent: {self.incumbent:.6f}")
            self.logger.info("=" * 100 + "\n")

        # Final LP relaxation check
        self.logger.debug("\n[Root] Final LP relaxation check...")
        self.cg_solver.master.solRelModel()

        lambda_list_root = {key: var.X for key, var in self.cg_solver.master.lmbda.items()}

        is_integral, lp_bound, most_frac_info = self.cg_solver.master.check_fractionality()

        # Update root node
        root_node = self.nodes[0]
        root_node.lp_bound = lp_bound
        root_node.is_integral = is_integral
        root_node.most_fractional_var = most_frac_info

        # Update root node's column pool
        self._update_node_column_pool(root_node)

        # Update node status
        if is_integral:
            root_node.status = 'integral'
            root_node.fathom_reason = 'integral'
            self.logger.info(f"\n✅ ROOT NODE IS INTEGRAL (LP)!")
        else:
            root_node.status = 'open'
            self.logger.warning(f"\n⚠️  ROOT NODE IS FRACTIONAL (LP)")

        self.logger.info(f"   LP Bound: {lp_bound:.6f}")
        self.logger.info(f"   Incumbent: {self.incumbent:.6f}" if self.incumbent < float('inf') else "   Incumbent: None")

        # Update global bounds
        self.best_lp_bound = lp_bound
        self.update_gap()

        # Save initial Root-LP/SOL
        self.cg_solver.master.Model.write('LPs/MP/LPs/master_node_root.lp')
        self.cg_solver.master.Model.write('LPs/MP/SOLs/master_node_root.sol')

        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" ROOT NODE SOLVED ".center(100, "="))
        self.logger.info(f"{'=' * 100}\n")

        return lp_bound, is_integral, most_frac_info, lambda_list_root


    def _compute_final_incumbent(self):
        """
        Compute incumbent from final RMP after CG convergence.

        This is the default behavior (when early_incumbent_iteration = 0).
        """
        master = self.cg_solver.master
        success = self._solve_rmp_as_ip(master, context="Final Incumbent")

        if success:
            self.logger.info(f"\n{'=' * 100}")
            self.logger.info("✅ FINAL INCUMBENT FOUND ".center(100, "="))
            self.logger.info(f"{'=' * 100}")
            self.logger.info(f"IP Objective:     {self.incumbent:.6f}")
            self.logger.info(f"LP Bound (root):  {master.Model.objBound:.6f}" if hasattr(master.Model, 'objBound') else "")
            self.logger.info(f"Gap:              {self.gap:.4%}")
            self.logger.info(f"{'=' * 100}\n")
        else:
            self.logger.warning(f"\n⚠️  Could not compute final incumbent")

    def _solve_rmp_as_ip(self, master, context="IP Solve"):
        """
        Solve the Restricted Master Problem as Integer Program.

        This is a helper method used by both early and final incumbent computation.

        Args:
            master: MasterProblem_d instance
            context: String describing the context (for logging)

        Returns:
            bool: True if successful and incumbent was updated
        """
        self.logger.info("=" * 100)
        self.logger.info(f"{context}: Solving RMP as Integer Program...".center(100))
        self.logger.info("=" * 100 + "\n")
        self.stats['ip_solves'] += 1

        try:
            # Save current variable types
            original_vtypes = {}
            for var in master.lmbda.values():
                original_vtypes[var.VarName] = var.VType
                var.VType = gu.GRB.INTEGER

            # Solve as IP
            master.Model.Params.OutputFlag = 1
            master.Model.Params.TimeLimit = 300  # 5 minute time limit
            master.Model.update()

            self.logger.info(f"[{context}] Starting optimization...")
            master.Model.optimize()

            success = False
            result_obj = float('inf')

            # Check solution status
            if master.Model.status == gu.GRB.OPTIMAL:
                ip_obj = master.Model.objVal

                # Update incumbent if better
                if ip_obj < self.incumbent:
                    self.incumbent = ip_obj
                    self.incumbent_solution = master.finalDicts(
                        self.cg_solver.global_solutions,
                        self.cg_solver.app_data, None
                    )
                    lambda_assignments = {}
                    for (p, a), var in master.lmbda.items():
                        if var.X > 0:
                            lambda_assignments[(p, a)] = var.X
                            print(f"Benesss ({p,a}): {var.X}")

                    self.incumbent_lambdas = lambda_assignments

                    self.stats['incumbent_updates'] += 1
                    self.update_gap()

                    self.logger.info(f"\n✅ New incumbent found: {self.incumbent:.6f}")
                    self.logger.info(f"   Gap: {self.gap:.4%}\n")

                    success = True
                    result_obj = ip_obj
                else:
                    self.logger.warning(f"\n⚠️  IP solution not better than current incumbent")
                    self.logger.info(f"   IP Objective:      {ip_obj:.6f}")
                    self.logger.info(f"   Current Incumbent: {self.incumbent:.6f}\n")
                    success = False
                    result_obj = ip_obj

            elif master.Model.status == gu.GRB.TIME_LIMIT:
                self.logger.warning(f"\n⚠️  IP solve hit time limit")
                if master.Model.SolCount > 0:
                    ip_obj = master.Model.objVal
                    self.logger.info(f"   Best found solution: {ip_obj:.6f}")
                    if ip_obj < self.incumbent:
                        self.incumbent = ip_obj
                        self.incumbent_solution = master.finalDicts(
                            self.cg_solver.global_solutions,
                            self.cg_solver.app_data, None
                        )
                        lambda_assignments = {}
                        for (p, a), var in master.lmbda.items():
                            if var.X > 0:
                                lambda_assignments[(p, a)] = int(round(var.X))
                        self.incumbent_lambdas = lambda_assignments

                        self.stats['incumbent_updates'] += 1
                        self.update_gap()
                        self.logger.info(f"   Updated incumbent: {self.incumbent:.6f}\n")
                        success = True
                        result_obj = ip_obj
                    else:
                        success = False
                        result_obj = ip_obj
                else:
                    self.logger.info(f"   No feasible solution found within time limit\n")
                    success = False
                    result_obj = float('inf')

            else:
                self.logger.error(f"❌ IP solve unsuccessful (status={master.Model.status})")
                success = False
                result_obj = float('inf')

            # Restore continuous relaxation
            for var in master.lmbda.values():
                var.VType = original_vtypes[var.VarName]

            master.Model.Params.OutputFlag = 0
            master.Model.Params.TimeLimit = float('inf')
            master.Model.update()

            return success

        except Exception as e:
            self.logger.error(f"❌ Error during {context}: {e}\n")

            # Restore original variable types
            for var in master.lmbda.values():
                if var.VarName in original_vtypes:
                    var.VType = original_vtypes[var.VarName]

            master.Model.Params.OutputFlag = 0
            master.Model.update()

            return False

    def should_fathom(self, node, lambdas_dict):
        """
        Determine if a node should be fathomed.

        A node is fathomed if:
        1. Its LP solution is integral (we found an IP solution)
        2. Its LP is infeasible
        3. Its LP bound >= incumbent (no better solution possible)

        Args:
            node: BnPNode to check

        Returns:
            bool: True if node should be fathomed
        """
        # Check 1: Integral solution
        if node.is_integral:
            node.status = 'fathomed'
            node.fathom_reason = 'integral'

            # Update incumbent if this is better
            if node.lp_bound < self.incumbent:
                print(f"\n✅ Node {node.node_id} found improving integral solution!")
                print(f"   Previous incumbent: {self.incumbent:.6f}")
                print(f"   New incumbent:      {node.lp_bound:.6f}")

                self.incumbent = node.lp_bound
                self.incumbent_solution = self.cg_solver.master.finalDicts(
                    self.cg_solver.global_solutions,
                    self.cg_solver.app_data, lambdas_dict
                )

                self.incumbent_lambdas = {k: int(v) for k, v in lambdas_dict.items() if v > 0}
                self.stats['incumbent_updates'] += 1
                self.update_gap()

                self.logger.info(f"   New gap: {self.gap:.4%}\n")

            return True

        # Check 2: Infeasible
        if node.lp_bound == float('inf'):
            node.status = 'fathomed'
            node.fathom_reason = 'infeasible'
            self.logger.info(f"   Node {node.node_id} fathomed: LP infeasible")
            return True

        # Check 3: Bound worse than incumbent
        if node.lp_bound >= self.incumbent - 1e-5:
            node.status = 'fathomed'
            node.fathom_reason = 'bound'
            self.logger.info(f"   Node {node.node_id} fathomed by bound: "
                        f"LP={node.lp_bound:.6f} >= UB={self.incumbent:.6f}")
            return True

        # Check 4:
        if node.status == 'fathomed':
            self._check_and_fathom_parents(node.node_id)
            return True

        # Node cannot be fathomed
        return False

    def update_gap(self):
        """
        Calculate optimality gap: (UB - LB) / UB.
        """
        if self.incumbent < float('inf') and self.best_lp_bound < float('inf'):
            if abs(self.incumbent) > 1e-10:
                self.gap = (self.incumbent - self.best_lp_bound) / abs(self.incumbent)
            else:
                self.gap = abs(self.incumbent - self.best_lp_bound)
        else:
            self.gap = float('inf')

    def solve(self, time_limit=3600, max_nodes=10000):
        """
        Main solve method for Branch-and-Price with full tree exploration.
        """
        self.start_time = time.time()

        self.logger.info("\n" + "=" * 100)
        self.logger.info(" BRANCH-AND-PRICE SOLVE ".center(100, "="))
        self.logger.info("=" * 100)
        self.logger.info(f"Time limit: {time_limit}s")
        self.logger.info(f"Max nodes: {max_nodes}")
        self.logger.info(f"Branching strategy: {self.branching_strategy.upper()}")
        self.logger.info("=" * 100 + "\n")

        if self.ip_heuristic_frequency > 0:
            self.logger.info(f"IP heuristic: Every {self.ip_heuristic_frequency} nodes")
        else:
            self.logger.info(f"IP heuristic: Disabled")

        # ========================================
        # PHASE 1: CREATE AND SOLVE ROOT NODE
        # ========================================
        root_node = self.create_root_node()
        # For 'bfs', open_nodes contains a tuple, so we extract the ID
        if self.search_strategy == 'bfs':
            self.open_nodes.pop()  # Remove placeholder

        lp_bound, is_integral, frac_info, root_lambdas = self.solve_root_node()
        self.stats['nodes_explored'] = 1

        # Add root node to open list with its solved bound for 'bfs'
        if self.search_strategy == 'bfs' and not is_integral:
            self.open_nodes.append((lp_bound, 0))

        # Check if root can be fathomed
        if self.should_fathom(root_node, None):
            self.logger.info(f"✅ Root node fathomed: {root_node.fathom_reason}")
            self.logger.info(f"   Solution is optimal!\n")
            self.stats['nodes_fathomed'] = 1
            if self.open_nodes:
                self.open_nodes.pop()
            self._finalize_and_print_results()
            return self._get_results_dict()

        # Root needs branching
        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" ROOT NODE REQUIRES BRANCHING ".center(100, "="))
        self.logger.info(f"{'=' * 100}\n")

        # Remove root from open nodes before branching
        if self.open_nodes:
            self.open_nodes.pop()

        # Branch on root
        branching_type, branching_info = self.select_branching_candidate(root_node, root_lambdas)
        print(branching_info, branching_type)

        if not branching_type:
            self.logger.warning(f"⚠️  Could not find branching candidate despite fractional solution!")
            self._finalize_and_print_results()
            return self._get_results_dict()

        # Create child nodes
        if branching_type == 'mp':
            left_child, right_child = self.branch_on_mp_variable(root_node, branching_info)
        else:  # 'sp'
            left_child, right_child = self.branch_on_sp_variable(root_node, branching_info)
        print(left_child, right_child)
        # Mark root as branched
        root_node.status = 'branched'
        self.stats['nodes_branched'] += 1

        # ========================================
        # PHASE 2: MAIN BRANCH-AND-PRICE LOOP
        # ========================================
        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" MAIN BRANCH-AND-PRICE LOOP ".center(100, "="))
        self.logger.info(f"{'=' * 100}\n")

        iteration = 0

        while self.open_nodes and iteration < max_nodes:
            iteration += 1

            # Check time limit
            elapsed = time.time() - self.start_time
            if elapsed > time_limit:
                self.logger.info(f"\n⏱️  Time limit reached: {elapsed:.2f}s > {time_limit}s")
                break

            # ========================================
            # PERIODIC IP HEURISTIC
            # ========================================
            # Run BEFORE processing the next node
            if self.ip_heuristic_frequency > 0 and iteration > 1:
                improved = self._run_ip_heuristic(iteration)

                # If incumbent improved significantly and no more open nodes, we're done
                if improved and not self.open_nodes:
                    self.logger.info("\n✅ All nodes fathomed after IP heuristic improvement!")
                    break

            # If all nodes fathomed, terminate
            if not self.open_nodes:
                break

            # ========================================
            # SELECT NEXT NODE
            # ========================================
            if self.search_strategy == 'bfs':
                # Best-first: sort by bound (ascending) and pop the best (lowest bound)
                # We sort descending and pop from the end for efficiency (O(1))
                sorted_open_nodes = sorted(self.open_nodes, key=lambda x: x[0], reverse=True)

                # Log the decision process
                decision_log_entry = {
                    'iteration': iteration,
                    'open_nodes_state': copy.deepcopy(sorted_open_nodes),
                    'chosen_node_id': sorted_open_nodes[-1][1],
                    'chosen_node_bound': sorted_open_nodes[-1][0]
                }
                self.stats['bfs_decision_log'].append(decision_log_entry)

                bound, current_node_id = sorted_open_nodes.pop()
                self.open_nodes = sorted_open_nodes  # update the list
                self.logger.info(f"   [BFS] Selected Node {current_node_id} with initial bound {bound:.4f}")

            else:  # DFS (default)
                current_node_id = self.open_nodes.pop()

            # Log the processing order for all strategies
            self.stats['node_processing_order'].append(current_node_id)

            self.logger.info(f"   🔎 Open nodes: {self.open_nodes}")
            current_node = self.nodes[current_node_id]

            print(f"\n{'╔' + '═' * 98 + '╗'}")
            print(f"║{f' PROCESSING NODE {current_node_id} (Iteration {iteration}) ':^98}║")
            print(f"║{f' Path: {current_node.path}, Depth: {current_node.depth} ':^98}║")
            print(f" Open Nodes: {len(self.open_nodes)}, Explored: {self.stats['nodes_explored']} ")
            print(f"║{f' Incumbent: {self.incumbent:.4f}, Best LB: {self.best_lp_bound:.4f} ':^98}║")
            print(f"{'╚' + '═' * 98 + '╝'}\n")

            # ========================================
            # SOLVE NODE WITH COLUMN GENERATION
            # ========================================
            try:
                lp_bound, is_integral, most_frac_info, node_lambdas = self.solve_node_with_cg(
                    current_node, max_cg_iterations=50
                )
            except Exception as e:
                self.logger.error(f"❌ Error solving node {current_node_id}: {e}")
                current_node.status = 'fathomed'
                current_node.fathom_reason = 'error'
                self.stats['nodes_fathomed'] += 1
                continue

            self.stats['nodes_explored'] += 1

            # Update best LP bound (for gap calculation)
            if lp_bound < self.best_lp_bound:
                self.best_lp_bound = lp_bound
                self.update_gap()

            # ========================================
            # CHECK FATHOMING
            # ========================================
            if self.should_fathom(current_node, node_lambdas):
                self.logger.info(f"✅ Node {current_node_id} fathomed: {current_node.fathom_reason}")
                self.stats['nodes_fathomed'] += 1

                # Print current status
                self.logger.info(f"\n   Status after fathoming:")
                self.logger.info(f"   ├─ Best LB: {self.best_lp_bound:.6f}")
                self.logger.info(f"   ├─ Incumbent: {self.incumbent:.6f}" if self.incumbent < float(
                    'inf') else "   ├─ Incumbent: None")
                self.logger.info(f"   ├─ Gap: {self.gap:.4%}" if self.gap < float('inf') else "   ├─ Gap: ∞")
                self.logger.info(f"   └─ Open nodes: {len(self.open_nodes)}\n")

                continue

            # ========================================
            # NODE NOT FATHOMED → BRANCH
            # ========================================
            self.logger.warning(f"\n⚠️  Node {current_node_id} requires branching (LP is fractional)")

            # Select branching candidate
            branching_type, branching_info = self.select_branching_candidate(current_node, node_lambdas)

            if not branching_type:
                self.logger.error(f"❌ Could not find branching candidate at node {current_node_id}")
                self.logger.error(f"   Marking as fathomed (should not happen!)")
                current_node.status = 'fathomed'
                current_node.fathom_reason = 'no_branching_candidate'
                self.stats['nodes_fathomed'] += 1
                continue

            # Create child nodes
            if branching_type == 'mp':
                left_child, right_child = self.branch_on_mp_variable(current_node, branching_info)
            else:  # 'sp'
                left_child, right_child = self.branch_on_sp_variable(current_node, branching_info)

            # Mark current node as branched
            current_node.status = 'branched'
            self.stats['nodes_branched'] += 1

            self.logger.info(f"\n✅ Created child nodes:")
            self.logger.info(f"   ├─ Left:  Node {left_child.node_id} (path: '{left_child.path}')")
            self.logger.info(f"   └─ Right: Node {right_child.node_id} (path: '{right_child.path}')")
            self.logger.info(f"\n   Open nodes queue: {self.open_nodes}")

        # ========================================
        # FINALIZATION
        # ========================================
        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" BRANCH-AND-PRICE TERMINATED ".center(100, "="))
        self.logger.info(f"{'=' * 100}")

        # Determine termination reason
        if not self.open_nodes:
            self.logger.info(f"✅ All nodes explored - Tree complete!")
        elif iteration >= max_nodes:
            self.logger.warning(f"⚠️  Node limit reached: {iteration} >= {max_nodes}")
            self.logger.warning(f"   {len(self.open_nodes)} nodes remain open")
        else:
            self.logger.info(f"⏱️  Time limit reached")
            self.logger.info(f"   {len(self.open_nodes)} nodes remain open")

        self._finalize_and_print_results()
        return self._get_results_dict()

    def _print_final_results(self):
        """Print final results."""
        self.logger.info("\n" + "=" * 100)
        self.logger.info(" BRANCH-AND-PRICE RESULTS ".center(100, "="))
        self.logger.info("=" * 100)
        self.logger.info(f"Status: Phase 1 Complete (Root Node Only)")
        self.logger.info(f"")
        self.logger.info(f"Bounds:")
        self.logger.info(f"  LP Bound (LB):  {self.best_lp_bound:.6f}")
        self.logger.info(
            f"  Incumbent (UB): {self.incumbent:.6f}" if self.incumbent < float('inf') else "  Incumbent (UB): None")
        self.logger.info(f"  Gap:            {self.gap:.4%}" if self.gap < float('inf') else "  Gap:            ∞")
        self.logger.info(f"")
        self.logger.info(f"Statistics:")
        self.logger.info(f"  Nodes Explored:   {self.stats['nodes_explored']}")
        self.logger.info(f"  Nodes Fathomed:   {self.stats['nodes_fathomed']}")
        self.logger.info(f"  CG Iterations:    {self.stats['total_cg_iterations']}")
        self.logger.info(f"  IP Solves:        {self.stats['ip_solves']}")
        self.logger.info(f"  Incumbent Updates: {self.stats['incumbent_updates']}")
        self.logger.info(f"  Total Time:       {self.stats['total_time']:.2f}s")
        self.logger.info(f"")
        self.logger.info(f"Root Node Info:")
        root = self.nodes[0]
        self.logger.info(f"  Status:         {root.status}")
        self.logger.info(f"  Is Integral:    {root.is_integral}")
        self.logger.info(f"  LP Bound:       {root.lp_bound:.6f}")
        if root.most_fractional_var:
            frac = root.most_fractional_var
            self.logger.info(
                f"  Most Frac Var:  {frac['var_name']} = {frac['value']:.6f} (dist={frac['fractionality']:.6f})")
        if root.fathom_reason:
            self.logger.info(f"  Fathom Reason:  {root.fathom_reason}")
        self.logger.info("=" * 100 + "\n")

    def _get_results_dict(self):
        """Create results dictionary."""
        return {
            'lp_bound': self.best_lp_bound,
            'incumbent': self.incumbent if self.incumbent < float('inf') else None,
            'gap': self.gap if self.gap < float('inf') else None,
            'is_integral': self.nodes[0].is_integral,
            'nodes_explored': self.stats['nodes_explored'],
            'nodes_fathomed': self.stats['nodes_fathomed'],
            'cg_iterations': self.stats['total_cg_iterations'],
            'ip_solves': self.stats['ip_solves'],
            'incumbent_updates': self.stats['incumbent_updates'],
            'total_time': self.stats['total_time'],
            'root_node': self.nodes[0]
        }

    # ============================================================================
    # BRANCHING LOGIC
    # ============================================================================

    def select_branching_candidate(self, node, node_lambda):
        """
        Select the most fractional variable for branching.

        Strategy depends on self.branching_strategy:
        - 'mp': Branch on Lambda_{na} (master variable)
        - 'sp': Branch on x_{njt} via beta_{njt} (subproblem variable)

        Tie-breaking: smallest n, then smallest a/j/t

        Args:
            node: BnPNode to select branching candidate from

        Returns:
            tuple: (branching_type, branching_info) or (None, None) if no fractional var
        """
        if self.branching_strategy == 'mp':
            return self._select_mp_branching_candidate(node)
        elif self.branching_strategy == 'sp':
            return self._select_sp_branching_candidate(node, node_lambda)
        else:
            raise ValueError(f"Unknown branching strategy: {self.branching_strategy}")

    def _select_mp_branching_candidate(self, node):
        """
        Select most fractional Lambda_{na} for MP branching.

        Returns:
            tuple: ('mp', branching_info) or (None, None)
        """
        if node.most_fractional_var is None:
            return None, None

        frac_info = node.most_fractional_var
        var_name = frac_info['var_name']

        if 'lmbda' not in var_name:
            self.logger.warning(f"⚠️  Unknown variable type: {var_name}")
            return None, None

        # Parse Lambda[n,a]
        parts = var_name.split('[')[1].split(']')[0].split(',')
        n = int(parts[0])
        a = int(parts[1])

        branching_info = {
            'profile': n,
            'column': a,
            'value': frac_info['value'],
            'floor': frac_info['floor'],
            'ceil': frac_info['ceil'],
            'fractionality': frac_info['fractionality']
        }

        return 'mp', branching_info

    def _select_sp_branching_candidate(self, node, lambdas):
        """
        Select most fractional beta_{njt} for SP branching.

        Uses node.column_pool to get correct column_ids instead of master.all_schedules.

        beta_{njt} = sum_{a: chi^a_{njt}=1} Lambda_{na}

        Returns:
            tuple: ('sp', branching_info) or (None, None)
        """
        beta_values = {}


        self.logger.info(f"\n[SP Branching] Computing beta values from node.column_pool...")
        self.logger.info(f"  Column pool size: {len(node.column_pool)}")
        self.logger.info(f"  Lambda values size: {len(lambdas)}")

        if len(node.column_pool) != len(lambdas):
            self.logger.warning(f"  ⚠️  Lambda pool is not equal sized as the column pool")

        # Iterate over Lambda variables to get their current LP values
        print(lambdas)
        for (n, a), lambda_val in lambdas.items():

            # lambda_val ist bereits ein Float - kein .X mehr nötig!
            if lambda_val < 1e-6:
                continue

            # Get column data from node's column pool
            if (n, a) not in node.column_pool:
                self.logger.warning(f"  ⚠️  Lambda[{n},{a}] = {lambda_val:.4f} but column not in pool!")
                continue

            col_data = node.column_pool[(n, a)]
            schedules_x = col_data.get('schedules_x', {})

            if not schedules_x:
                continue

            # Extract assignments from this column
            for (p, j, t, _), chi_val in schedules_x.items():
                if p == n and chi_val > 0.5:
                    key = (n, j, t)
                    beta_values[key] = beta_values.get(key, 0.0) + lambda_val

        self.logger.info(f"  Found {len(beta_values)} non-zero beta values")

        # Find most fractional beta
        best_candidate = None
        max_fractionality = 0.0

        for (n, j, t), beta_val in beta_values.items():
            floor_val = int(beta_val)
            ceil_val = floor_val + 1

            dist_to_floor = beta_val - floor_val
            dist_to_ceil = ceil_val - beta_val
            fractionality = min(dist_to_floor, dist_to_ceil)

            if fractionality > 1e-5:  # Fractional
                is_better = False

                if fractionality > max_fractionality + 1e-10:
                    is_better = True
                elif abs(fractionality - max_fractionality) < 1e-10:
                    # Tie: prefer smaller n, then j, then t
                    if best_candidate is None:
                        is_better = True
                    else:
                        current = (n, j, t)
                        best = best_candidate['assignment']
                        if current < best:
                            is_better = True

                if is_better:
                    max_fractionality = fractionality
                    best_candidate = {
                        'profile': n,
                        'agent': j,
                        'period': t,
                        'beta_value': beta_val,
                        'floor': floor_val,
                        'ceil': ceil_val,
                        'fractionality': fractionality,
                        'assignment': (n, j, t)
                    }

        print('Best Beta', best_candidate)

        if best_candidate is None:
            self.logger.error(f"  ❌ No fractional beta found!")

            return None, None

        self.logger.info(f"\n  ✅ Most fractional beta:")
        self.logger.info(f"     beta[{best_candidate['profile']},{best_candidate['agent']},{best_candidate['period']}] = {best_candidate['beta_value']:.6f}")
        self.logger.info(f"     Fractionality: {best_candidate['fractionality']:.6f}")
        self.logger.info(f"     Floor: {best_candidate['floor']}, Ceil: {best_candidate['ceil']}")

        return 'sp', best_candidate

    def branch_on_mp_variable(self, parent_node, branching_info):
        """
        Branch on Master Problem Variable Lambda_{na}.

        Creates two child nodes:
        - Left:  Lambda_{na} <= floor(Lambda_hat)
        - Right: Lambda_{na} >= ceil(Lambda_hat)

        Paper Section 3.2.4, Equation (branch_mp1)

        Args:
            parent_node: BnPNode to branch from
            branching_info: Dict with 'profile', 'column', 'value', 'floor', 'ceil'

        Returns:
            tuple: (left_child, right_child) - two new BnPNode objects
        """
        n = branching_info['profile']
        a = branching_info['column']
        lambda_value = branching_info['value']
        floor_val = branching_info['floor']
        ceil_val = branching_info['ceil']

        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(f" BRANCHING ON MP VARIABLE ".center(100, "="))
        self.logger.info(f"{'=' * 100}")
        self.logger.info(f"Branching on Lambda[{n},{a}] = {lambda_value:.6f}")
        self.logger.info(f"  Left:  Lambda[{n},{a}] <= {floor_val}")
        self.logger.info(f"  Right: Lambda[{n},{a}] >= {ceil_val}")

        # Get original schedule for no-good cut
        original_schedule = None
        if (n, a) in parent_node.column_pool:
            original_schedule = parent_node.column_pool[(n, a)].get('schedules_x', {})
            self.logger.info(f"\n  ✅ Found column ({n},{a}) in parent's column pool")
            self.logger.info(f"     Schedule has {len(original_schedule)} assignments")

            # Show first few assignments
            if original_schedule:
                sample_assignments = list(original_schedule.items())[:3]
                for key, val in sample_assignments:
                    self.logger.info(f"       {key}: {val}")
        else:
            self.logger.error(f"\n  ❌ ERROR: Column ({n},{a}) NOT found in parent's column pool!")
            self.logger.error(
                f"     Available columns for profile {n}: {[col_id for (p, col_id) in parent_node.column_pool.keys() if p == n]}")
            self.logger.error(f"     No-good cut cannot be added!")

        # -------------------------
        # LEFT CHILD
        # -------------------------
        self.node_counter += 1
        left_child = BnPNode(
            node_id=self.node_counter,
            parent_id=parent_node.node_id,
            depth=parent_node.depth + 1,
            path=parent_node.path + 'l'
        )

        # Create left branching constraint
        from branching_constraints import MPVariableBranching

        left_constraint = MPVariableBranching(
            profile_n=n,
            column_a=a,
            bound=floor_val,
            direction='left',
            original_schedule=original_schedule
        )

        # Inherit branching constraints from parent + add new one
        left_child.branching_constraints = parent_node.branching_constraints.copy()
        left_child.branching_constraints.append(left_constraint)

        # Inherit compatible columns from parent
        self._inherit_columns_from_parent(left_child, parent_node)

        # -------------------------
        # RIGHT CHILD
        # -------------------------
        self.node_counter += 1
        right_child = BnPNode(
            node_id=self.node_counter,
            parent_id=parent_node.node_id,
            depth=parent_node.depth + 1,
            path=parent_node.path + 'r'
        )

        # Create right branching constraint
        right_constraint = MPVariableBranching(
            profile_n=n,
            column_a=a,
            bound=ceil_val,
            direction='right'
        )

        # Inherit branching constraints from parent + add new one
        right_child.branching_constraints = parent_node.branching_constraints.copy()
        right_child.branching_constraints.append(right_constraint)

        # Inherit all columns (no restriction on right branch)
        self._inherit_columns_from_parent(right_child, parent_node)

        # -------------------------
        # EVALUATE AND STORE NODES
        # -------------------------
        if self.search_strategy == 'bfs':
            # Solve initial LP to get a bound for node selection
            self.logger.info("\n  [BFS] Evaluating child nodes...")
            left_bound = self._solve_initial_lp(left_child)
            right_bound = self._solve_initial_lp(right_child)
            left_child.lp_bound = left_bound
            right_child.lp_bound = right_bound
            self.logger.info(f"    - Left Child (Node {left_child.node_id}) initial LP bound: {left_bound:.4f}")
            self.logger.info(f"    - Right Child (Node {right_child.node_id}) initial LP bound: {right_bound:.4f}")

        # Store nodes in the main dictionary
        self.nodes[left_child.node_id] = left_child
        self.nodes[right_child.node_id] = right_child

        # Add to open nodes list based on the search strategy
        if self.search_strategy == 'bfs':
            if left_child.lp_bound < float('inf'):
                self.open_nodes.append((left_child.lp_bound, left_child.node_id))
            if right_child.lp_bound < float('inf'):
                self.open_nodes.append((right_child.lp_bound, right_child.node_id))
        else:  # DFS
            self.open_nodes.append(right_child.node_id)
            self.open_nodes.append(left_child.node_id)

        # Update parent status
        parent_node.status = 'branched'

        self.logger.info(f"  Created left child:  Node {left_child.node_id} (depth {left_child.depth})")
        self.logger.info(f"  Created right child: Node {right_child.node_id} (depth {right_child.depth})")
        self.logger.info(f"{'=' * 100}\n")

        self.stats['nodes_branched'] += 1

        return left_child, right_child

    def _solve_initial_lp(self, node):
        """
        Solves the master problem for a node once as an LP without CG.
        This provides a quick initial lower bound for the 'bfs' strategy.
        """
        master = self._build_master_for_node(node)
        master.Model.setParam('OutputFlag', 0)  # Suppress Gurobi output for this solve
        master.solRelModel()
        master.Model.setParam('OutputFlag', 1)  # Re-enable for subsequent solves

        if master.Model.status == gu.GRB.OPTIMAL:
            return master.Model.ObjVal
        else:
            # If infeasible or unbounded, return infinity
            return float('inf')

    def _inherit_columns_from_parent(self, child_node, parent_node):
        """
        Inherit columns from parent to child node.

        Filters out columns that are incompatible with the child's
        branching constraints.

        Args:
            child_node: BnPNode receiving columns
            parent_node: BnPNode providing columns
        """
        child_node.column_pool = {}
        inherited_count = 0
        filtered_count = 0

        for (p, col_id), col_data in parent_node.column_pool.items():
            # Check compatibility with all branching constraints
            is_compatible = True

            for constraint in child_node.branching_constraints:
                if not constraint.is_column_compatible(col_data):
                    is_compatible = False
                    break

            if is_compatible:
                child_node.column_pool[(p, col_id)] = col_data.copy()
                inherited_count += 1
            else:
                filtered_count += 1

        self.logger.info(f"    Column inheritance: {inherited_count} inherited, {filtered_count} filtered")

    def solve_node_with_cg(self, node, max_cg_iterations=100):
        """
        Solve a node using Column Generation with branching constraints.

        This performs full CG at a node:
        1. Build master with inherited columns + branching constraints
        2. CG loop: solve LP, get duals, price subproblems, add columns
        3. Check integrality
        4. Return LP bound

        Args:
            node: BnPNode to solve
            max_cg_iterations: Maximum CG iterations at this node

        Returns:
            tuple: (lp_bound, is_integral, most_frac_info)
        """
        self.logger.info(f"\n{'─' * 100}")
        self.logger.info(f" SOLVING NODE {node.node_id} (path: '{node.path}', depth {node.depth}) ".center(100, "─"))
        self.logger.info(f"{'─' * 100}")
        self.logger.info(f"Branching constraints: {len(node.branching_constraints)}")
        self.logger.info(f"Column pool size: {len(node.column_pool)}")

        # Show column distribution
        cols_per_profile = {}
        for (p, _) in node.column_pool.keys():
            cols_per_profile[p] = cols_per_profile.get(p, 0) + 1
        self.logger.info(f"Columns per profile (sample): {dict(list(cols_per_profile.items())[:3])}")
        self.logger.info(f"{'─' * 100}\n")

        # 1. Build master problem and save LP for this node
        master = self._build_master_for_node(node)
        master.Model.update()


        # Determine branching profile (from constraints)
        branching_profile = self._get_branching_profile(node)
        if branching_profile:
            self.logger.info(f"    [SP Saving] Branching profile: {branching_profile}")

        # 2. Column Generation loop
        threshold = self.cg_solver.threshold  # Use same threshold as CG
        cg_iteration = 0

        # Node time limit
        node_start_time = time.time()
        NODE_TIME_LIMIT = 300

        self.logger.debug(f"\n    [Debug] Constraints BEFORE CG loop:")
        for c in master.Model.getConstrs():
            if 'sp_branch' in c.ConstrName:
                self.logger.debug(f"      {c.ConstrName}: {c.Sense} {c.RHS}")

        while cg_iteration < max_cg_iterations:
            if time.time() - node_start_time > NODE_TIME_LIMIT:
                self.logger.debug(f"⏱️  Node {node.node_id} time limit reached")
                break

            cg_iteration += 1

            self.logger.info(f"    [CG Iter {cg_iteration}] Solving master LP...")

            # Solve master as LP
            master.solRelModel()
            if master.Model.status != 2:  # GRB.OPTIMAL
                self.logger.warning(f"    ⚠️  Master in CG-iterations infeasible or unbounded at node {node.node_id}")
                return float('inf'), False, None, {}

            current_lp_obj = master.Model.objVal
            self.logger.info(f"    [CG Iter {cg_iteration}] LP objective: {current_lp_obj:.6f}")

            # Get duals from master
            duals_pi, duals_gamma = master.getDuals()
            self.logger.info(self.branching_strategy)

            # Get branching constraint duals if SP-branching is used
            branching_duals = {}
            if self.branching_strategy == 'sp':
                branching_duals = self._get_branching_constraint_duals(master, node)

            # 3. Solve subproblems for all profiles
            new_columns_found = False
            columns_added_this_iter = 0

            for profile in self.cg_solver.P_Join:
                # Build and solve subproblem with branching constraints
                sp = self._build_subproblem_for_node(
                    profile, node, duals_pi, duals_gamma, branching_duals
                )
                # SAVE FIRST SP FOR BRANCHING PROFILE
                if profile == branching_profile:
                    sp_filename = f"LPs/SPs/pricing/sp_node_{node.node_id}_profile_{profile}_iter{cg_iteration}.lp"
                    sp.Model.write(sp_filename)
                    self.logger.info(f"    ✅ [SP Saved] First pricing SP for branching profile {profile}: {sp_filename}")
                sp.solModel()

                # Check reduced cost
                if sp.Model.status == 2 and sp.Model.objVal < -threshold:
                    self.logger.info(f'Red. cost for profile {profile} : {sp.Model.objVal}')

                    # Add column to node and master
                    self._add_column_from_subproblem(sp, profile, node, master)
                    new_columns_found = True
                    columns_added_this_iter += 1
                    master.Model.update()

            self.logger.info(f"    [CG Iter {cg_iteration}] Added {columns_added_this_iter} new columns")

            # Check convergence
            if not new_columns_found:
                self.logger.info(f"    [CG] Converged after {cg_iteration} iterations - no improving columns found")
                break
            master.Model.update()


        # 4. Final LP solve and integrality check
        self.logger.info(f"\n    [Node {node.node_id}] Final LP solve...")
        master.Model.write(f"LPs/MP/LPs/mp_final_{node.node_id}.lp")
        master.solRelModel()
        if master.Model.status != 2:  # GRB.OPTIMAL
            self.logger.warning(f"    ⚠️  Final Master infeasible or unbounded at node {node.node_id}")
            return float('inf'), False, None, {}

        if master.Model.status == 2:
            lambda_list_cg = {
                key: var.X for key, var in master.lmbda.items()
            }
        else:
            lambda_list_cg = {}

        master.Model.write(f"LPs/MP/SOLs/mp_node_{node.node_id}.sol")
        is_integral, lp_obj, most_frac_info = master.check_fractionality()

        if is_integral:
            self.logger.info(f"\n✅ INTEGRAL SOLUTION FOUND AT NODE {node.node_id}!")
            self.logger.info(f"   LP Bound: {lp_obj:.6f}")

        # Store results in node
        node.lp_bound = lp_obj
        node.is_integral = is_integral
        node.most_fractional_var = most_frac_info

        self.logger.info(f"\n    [Node {node.node_id}] Results:")
        self.logger.info(f"      LP Bound: {lp_obj:.6f}")
        self.logger.info(f"      Is Integral: {is_integral}")
        self.logger.info(f"      CG Iterations: {cg_iteration}")
        self.logger.info(f"      Final column pool: {len(node.column_pool)} columns")

        if most_frac_info:
            self.logger.info(f"      Most fractional: {most_frac_info['var_name']} = {most_frac_info['value']:.6f}")

        self.logger.info(f"{'─' * 100}\n")

        self.stats['total_cg_iterations'] += cg_iteration

        return lp_obj, is_integral, most_frac_info, lambda_list_cg

    def _build_master_for_node(self, node):
        """
        Build master problem for a node with inherited columns and branching constraints.
        """
        from masterproblem import MasterProblem_d

        self.logger.info(f"    [Master] Building master problem for node {node.node_id}...")

        # Create master
        master = MasterProblem_d(
            self.cg_solver.data,
            self.cg_solver.Max_t_cg,
            self.cg_solver.Nr_agg,
            self.cg_solver.Req_agg,
            self.cg_solver.pre_x,
            self.cg_solver.E_dict
        )

        # Build model with start sol (creates basic constraints)
        master.buildModel()
        master.startSol(self.start_x, self.start_los)
        master.Model.update()

        self.logger.info(f"    [Master] Basic model built with {len(master.Model.getConstrs())} constraints")

        # ✅ CRITICAL FIX: Add initial columns (col_id=1) to all_schedules for SP branching
        self.logger.info(f"    [Master] Adding initial columns (col_id=1) to all_schedules...")
        initial_cols_added = 0
        for (profile, col_id), col_data in node.column_pool.items():
            if col_id == 1:
                schedules_x = col_data.get('schedules_x', {})
                if schedules_x:
                    master.addSchedule(schedules_x)
                    initial_cols_added += 1

                    # Debug: Show first assignment
                    if initial_cols_added == 1:
                        sample_key = list(schedules_x.keys())[0]
                        self.logger.debug(f"      Sample initial column: {sample_key} = {schedules_x[sample_key]}")

        self.logger.info(f"    [Master] Added {initial_cols_added} initial columns to all_schedules")

        sp_branching_active = False

        # Load columns
        self.logger.info(f"    [Master] Loading {len(node.column_pool)} columns from pool...")

        for (profile, col_id), col_data in node.column_pool.items():

            if col_id >= 2:

                # Add schedules to master
                schedules_x = col_data.get('schedules_x', {})
                schedules_los = col_data.get('schedules_los', {})

                if not schedules_x:
                    self.logger.warning(f"      ⚠️ WARNING: Column ({profile},{col_id}) has empty schedules_x!")
                    continue

                master.addSchedule(schedules_x)
                master.addLOS(schedules_los)

                # Get pre-computed lists or create them
                x_list = col_data.get('x_list', list(schedules_x.values()))
                if profile in master.P_Post:
                    los_list = [0]
                elif profile in master.P_Focus and col_id >= 2:
                    los_list = col_data.get('los_list', list(schedules_los.values()))

                lambda_list = self._create_lambda_list(profile)

                # Build coefficient vector: [lambda_coefs, x_coefs]
                col_coefs = lambda_list + x_list

                # In MP branching, only variable bounds are set, no new constraints.
                if sp_branching_active:
                    branching_coefs = self._compute_branching_coefficients_for_column(
                        col_data, profile, col_id, node.branching_constraints
                    )
                    if all(x == 0 for x in branching_coefs):
                        self.logger.info(
                            f"      [Column with postive Chi {profile},{col_id}] Added {len(branching_coefs)} branching coefficients")
                        sys.exit()
                    col_coefs = col_coefs + branching_coefs
                    self.logger.info(
                        f"      [Column {profile},{col_id}] Added {len(branching_coefs)} branching coefficients")

                master.addLambdaVar(profile, col_id, col_coefs, los_list)
            master.Model.update()

        # Update Obj coefficients for initial column
        for (profile, col_id), col_data in node.column_pool.items():
            schedules_los = col_data.get('schedules_los', {})
            if profile in master.P_Join and col_id == 1:
                master.lmbda[profile, 1].Obj = col_data.get('los_list', schedules_los.values())[0]
        master.Model.update()

        # SP-Branching: adds new constraints → need coefficients
        # MP-Branching: only sets variable bounds → NO new coefficients needed
        if node.branching_constraints:
            self.logger.info(f"    [Master] Applying {len(node.branching_constraints)} branching constraints...")

            for constraint in node.branching_constraints:
                self.logger.info('Cons', constraint)

                constraint.apply_to_master(master, node)  # ← node übergeben!

                # Check if this is SP branching (adds constraints)
                if hasattr(constraint, 'master_constraint') and constraint.master_constraint is not None:
                    sp_branching_active = True

            master.Model.update()
            self.logger.info(f"    [Master] Now have {len(master.Model.getConstrs())} constraints")
            self.logger.info(f"    [Master] SP-Branching constraints added: {sp_branching_active}")

            # DEBUG EXIT
            if sp_branching_active and node.node_id > 0:
                master.Model.write(f"LPs/MP/LPs/master_branch_node_{node.node_id}.lp")

                # Show constraint details
                for c in master.Model.getConstrs():
                    if 'sp_branch' in c.ConstrName:
                        self.logger.info(f"  Constraint: {c.ConstrName}")


        self.logger.info(f"    [Master] Master problem ready:")
        self.logger.info(f"             - {len(master.lmbda)} lambda variables")
        self.logger.info(f"             - {len(master.Model.getConstrs())} constraints")

        return master

    def _compute_branching_coefficients_for_column(self, col_data, profile, col_id, branching_constraints):
        """
        Compute coefficients for branching constraints for an existing column.

        CRITICAL: A branching constraint on profile n only affects columns for profile n!
        """
        from branching_constraints import SPVariableBranching, MPVariableBranching

        coefs = []
        schedules_x = col_data.get('schedules_x', {})

        for constraint in branching_constraints:
            if isinstance(constraint, SPVariableBranching):
                n, j, t = constraint.profile, constraint.agent, constraint.period

                # ✅ CRITICAL FIX: If this column is not for the branched profile, coef = 0
                if profile != n:
                    coefs.append(0)
                    continue

                # Only if profile == n: check if this column has assignment (j,t)
                chi_value = 0
                for (p, j_sched, t_sched, a), val in schedules_x.items():
                    if p == profile and j_sched == j and t_sched == t and a == col_id:
                        chi_value = val
                        break

                coef = 1 if chi_value > 0.5 else 0
                coefs.append(coef)

            elif isinstance(constraint, MPVariableBranching):
                # MP branching uses variable bounds, not linear constraints
                coefs.append(0)

        return coefs

    def _create_lambda_list(self, profile):
        """
        Create lambda list for a profile (indicator vector).

        Args:
            profile: Profile index

        Returns:
            list: Lambda list with 1 at profile position, 0 elsewhere
        """
        if profile in self.cg_solver.P_Join:
            ind = self.cg_solver.P_Join.index(profile)
            lst = [0] * len(self.cg_solver.P_Join)
            lst[ind] = 1
            return lst
        return []

    def _get_branching_constraint_duals(self, master, node):
        """
        Extract dual variables from SP branching constraints.

        According to Paper Eq. (branch:sub4):
        - Left branch (≤): δ^L ≤ 0
        - Right branch (≥): δ^R ≥ 0
        - Both are ADDED in pricing: - sum(δ^L + δ^R)
        """
        branching_duals = {}

        self.logger.info(f"\n      [Extracting Branching Duals] Node {node.node_id}, Path: '{node.path}'")
        self.logger.info(f"      Total branching constraints: {len(node.branching_constraints)}")

        sp_constraints_found = 0

        for constraint in node.branching_constraints:
            # Only SP-Variable Branching constraints have master constraints
            if not hasattr(constraint, 'master_constraint') or constraint.master_constraint is None:
                continue

            try:
                dual_val = constraint.master_constraint.Pi
                sp_constraints_found += 1

                # Validate dual sign (according to constraint direction)
                if constraint.dir == 'left' and dual_val > 1e-6:
                    self.logger.warning(f"      ⚠️  WARNING: Left branch (≤) has positive dual: {dual_val:.6f}")
                if constraint.dir == 'right' and dual_val < -1e-6:
                    self.logger.warning(f"      ⚠️  WARNING: Right branch (≥) has negative dual: {dual_val:.6f}")

                # Store with unique key (profile, agent, period, level)
                key = (constraint.profile, constraint.agent, constraint.period, constraint.level)
                branching_duals[key] = dual_val

                self.logger.info(f"      [Dual] Level {constraint.level:2d} ({constraint.dir:5s}): "
                            f"x[{constraint.profile},{constraint.agent:2d},{constraint.period:2d}] "
                            f"→ π={dual_val:+.6f}")

            except Exception as e:
                self.logger.warning(f"      ⚠️  Could not extract dual from constraint: {e}")

        self.logger.info(f"      Found {sp_constraints_found} SP branching duals\n")
        return branching_duals

    def _build_subproblem_for_node(self, profile, node, duals_pi, duals_gamma, branching_duals=None):
        """
        Build subproblem for a profile at a node with branching constraints.

        Uses node-local column IDs and REAL duals from master LP.

        Args:
            profile: Profile index
            node: BnPNode
            duals_pi: Dual variables for capacity constraints
            duals_gamma: Dual variables for profile constraints
            branching_duals: Dict of branching constraint duals (for SP-branching)

        Returns:
            Subproblem: Subproblem with constraints
        """
        from subproblem import Subproblem

        if branching_duals is None:
            branching_duals = {}

        # Filter relevant branching duals for this profile
        relevant_duals = {key: value for key, value in branching_duals.items() if key[0] == profile}

        if relevant_duals:
            duals_delta = sum(relevant_duals.values())

            # Detailed logging
            self.logger.info(f"\n      [SP Duals] Profile {profile} has {len(relevant_duals)} branching duals:")
            for (p, j, t, level), dual_val in relevant_duals.items():
                self.logger.info(f"         Level {level}: x[{p},{j},{t}] → dual={dual_val:.6f}")
            self.logger.info(f"      [SP Duals] Total duals_delta = {duals_delta:.6f}\n")
        else:
            duals_delta = 0.0
            self.logger.info(f"      [SP Duals] Profile {profile} has NO branching constraints\n")

        # Determine next col_id based on column_pool of this node
        profile_columns = [col_id for (p, col_id) in node.column_pool.keys() if p == profile]

        if profile_columns:
            next_col_id = max(profile_columns) + 1
        else:
            next_col_id = 1

        # Create subproblem mit echten Duals
        sp = Subproblem(
            self.cg_solver.data,
            duals_gamma,
            duals_pi,
            duals_delta,
            profile,
            next_col_id,
            self.cg_solver.Req_agg,
            self.cg_solver.Entry_agg,
            self.cg_solver.app_data,
            self.cg_solver.W_coeff,
            self.cg_solver.E_dict,
            self.cg_solver.S_Bound,
            learn_method=self.cg_solver.learn_method,
            reduction=True,
            num_tangents=10,
            node_path=node.path
        )

        sp.buildModel()

        # Apply all branching constraints
        for constraint in node.branching_constraints:
            constraint.apply_to_subproblem(sp)

        sp.Model.update()
        return sp

    def _add_column_from_subproblem(self, subproblem, profile, node, master):
        """
        Add a column generated from a subproblem to node and master.

        CRITICAL: Must compute branching coefficients for SP-branching constraints!
        """
        col_id = subproblem.col_id

        # Extract solution from subproblem
        schedules_x, x_list, _ = subproblem.getOptVals('x')
        schedules_los, los_list, _ = subproblem.getOptVals('LOS')

        # Create column data
        col_data = {
            'index': profile,
            'column_id': col_id,
            'schedules_x': schedules_x,
            'schedules_los': schedules_los,
            'x_list': x_list,
            'los_list': los_list,
            'reduced_cost': subproblem.Model.objVal
        }

        # Add to node's column pool
        node.column_pool[(profile, col_id)] = col_data

        # Add to master
        master.addSchedule(schedules_x)
        master.addLOS(schedules_los)

        # Create coefficient lists
        lambda_list = self._create_lambda_list(profile)

        # Basic coefficients
        col_coefs = lambda_list + x_list

        # ========================================================================
        # ADD SP-BRANCHING COEFFICIENTS IF NEEDED
        # ========================================================================
        sp_branching_constraints = [c for c in node.branching_constraints
                                    if hasattr(c, 'master_constraint')
                                    and c.master_constraint is not None]

        print(col_data, profile, col_id, node.branching_constraints, sep="\n")

        if sp_branching_constraints:
            branching_coefs = self._compute_branching_coefficients_for_column(
                col_data, profile, col_id, node.branching_constraints
            )
            col_coefs = col_coefs + branching_coefs

            print(f'New_Coeffs for profile {profile} are: {branching_coefs}')

            self.logger.info(f"        [Column] Added {len(branching_coefs)} branching coefficients "
                        f"for new column ({profile}, {col_id})")

        # Verify length
        expected_length = len(master.Model.getConstrs())
        actual_length = len(col_coefs)
        print(expected_length, actual_length, sep="\n")


        if actual_length != expected_length:
            self.logger.error(f"        ❌ ERROR: Coefficient mismatch when adding new column!")
            self.logger.error(f"           Expected: {expected_length}, Got: {actual_length}")
            raise ValueError("Coefficient vector length mismatch!")

        # Add variable to master
        master.addLambdaVar(
            profile, col_id,
            col_coefs,
            los_list
        )

        self.logger.info(f"        [Column] Added column ({profile}, {col_id}) "
                    f"with reduced cost {subproblem.Model.objVal:.6f}")



    def branch_on_sp_variable(self, parent_node, branching_info):
        """
        Branch on Subproblem Variable x_{njt}.

        Creates two child nodes:
        - Left:  x_{njt} = 0  (no assignment)
        - Right: x_{njt} = 1  (force assignment)

        Paper Section 3.2.4, Equations (branch:sub2) and (branch:sub3)

        Args:
            parent_node: BnPNode to branch from
            branching_info: Dict with 'profile', 'agent', 'period', 'beta_value', etc.

        Returns:
            tuple: (left_child, right_child)
        """
        n = branching_info['profile']
        j = branching_info['agent']
        t = branching_info['period']
        beta_val = branching_info['beta_value']
        floor_val = branching_info['floor']
        ceil_val = branching_info['ceil']

        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(f" BRANCHING ON SP VARIABLE ".center(100, "="))
        self.logger.info(f"{'=' * 100}")
        self.logger.info(f"Branching on x[{n},{j},{t}], beta = {beta_val:.6f}")
        self.logger.info(f"  Left:  x[{n},{j},{t}] = 0")
        self.logger.info(f"  Right: x[{n},{j},{t}] = 1")

        # -------------------------
        # LEFT CHILD
        # -------------------------
        self.node_counter += 1
        left_child = BnPNode(
            node_id=self.node_counter,
            parent_id=parent_node.node_id,
            depth=parent_node.depth + 1,
            path=parent_node.path + 'l'
        )

        from branching_constraints import SPVariableBranching

        left_constraint = SPVariableBranching(
            profile_n=n,
            agent_j=j,
            period_t=t,
            dir='left',
            level=left_child.depth,
            floor_val=floor_val,
            ceil_val=ceil_val
        )

        left_child.branching_constraints = parent_node.branching_constraints.copy()
        left_child.branching_constraints.append(left_constraint)

        self._inherit_columns_from_parent(left_child, parent_node)

        # -------------------------
        # RIGHT CHILD
        # -------------------------
        self.node_counter += 1
        right_child = BnPNode(
            node_id=self.node_counter,
            parent_id=parent_node.node_id,
            depth=parent_node.depth + 1,
            path=parent_node.path + 'r'
        )

        right_constraint = SPVariableBranching(
            profile_n=n,
            agent_j=j,
            period_t=t,
            dir='right',
            level=right_child.depth,
            floor_val=floor_val,
            ceil_val=ceil_val
        )

        right_child.branching_constraints = parent_node.branching_constraints.copy()
        right_child.branching_constraints.append(right_constraint)

        self._inherit_columns_from_parent(right_child, parent_node)

        # -------------------------
        # EVALUATE AND STORE NODES
        # -------------------------
        if self.search_strategy == 'bfs':
            # Solve initial LP to get a bound for node selection
            self.logger.info("\n  [BFS] Evaluating child nodes...")
            left_bound = self._solve_initial_lp(left_child)
            right_bound = self._solve_initial_lp(right_child)
            left_child.lp_bound = left_bound
            right_child.lp_bound = right_bound
            self.logger.info(f"    - Left Child (Node {left_child.node_id}) initial LP bound: {left_bound:.4f}")
            self.logger.info(f"    - Right Child (Node {right_child.node_id}) initial LP bound: {right_bound:.4f}")

        # Store nodes in the main dictionary
        self.nodes[left_child.node_id] = left_child
        self.nodes[right_child.node_id] = right_child

        # Add to open nodes list based on the search strategy
        if self.search_strategy == 'bfs':
            if left_child.lp_bound < float('inf'):
                self.open_nodes.append((left_child.lp_bound, left_child.node_id))
            if right_child.lp_bound < float('inf'):
                self.open_nodes.append((right_child.lp_bound, right_child.node_id))
        else:  # DFS
            # Add to open nodes (DFS: right first, then left, so left is processed first)
            self.open_nodes.append(right_child.node_id)
            self.open_nodes.append(left_child.node_id)

        parent_node.status = 'branched'

        self.logger.info(f"  Created left child:  Node {left_child.node_id} (depth {left_child.depth})")
        self.logger.info(f"  Created right child: Node {right_child.node_id} (depth {right_child.depth})")
        self.logger.info(f"{'=' * 100}\n")

        self.stats['nodes_branched'] += 1

        return left_child, right_child

    def _update_node_column_pool(self, node):
        """
        Update node's column pool with all columns from CG solver's global_solutions.

        This should be called after solving a node with CG to ensure the node
        has all generated columns in its pool.

        Args:
            node: BnPNode to update
        """
        self.logger.info(f"\n[Column Pool] Updating node {node.node_id} with generated columns...")

        initial_count = len(node.column_pool)

        # Iterate over all columns in global_solutions['x']
        # This is the authoritative source for all generated columns
        for (p, col_id) in self.cg_solver.global_solutions.get('x', {}).keys():
            # Skip if already in pool
            if (p, col_id) in node.column_pool:
                continue

            # Extract column data from global_solutions
            col_data = {
                'index': p,
                'column_id': col_id,
            }

            # Get x variables (assignments) - this is schedules_x
            # Format: {(p, j, t, itr): value}
            x_solution = self.cg_solver.global_solutions['x'][(p, col_id)]
            col_data['schedules_x'] = x_solution.copy()

            # Get LOS
            if (p, col_id) in self.cg_solver.global_solutions.get('LOS', {}):
                los_solution = self.cg_solver.global_solutions['LOS'][(p, col_id)]
                col_data['schedules_los'] = los_solution.copy()
            else:
                col_data['schedules_los'] = {}

            # Get other solution variables
            for var_name in ['y', 'z', 'S', 'l']:
                if (p, col_id) in self.cg_solver.global_solutions.get(var_name, {}):
                    col_data[f'{var_name}_data'] = self.cg_solver.global_solutions[var_name][(p, col_id)]

            if 'App' in self.cg_solver.global_solutions:
                if (p, col_id) in self.cg_solver.global_solutions['App']:
                    col_data['App_data'] = self.cg_solver.global_solutions['App'][(p, col_id)]

            # Add to column pool
            node.column_pool[(p, col_id)] = col_data

        final_count = len(node.column_pool)
        added_count = final_count - initial_count

        self.logger.info(f"[Column Pool] Updated: {initial_count} → {final_count} columns (+{added_count} new)")

        # Debug: Show some schedules_x info
        if added_count > 0:
            sample_key = list(node.column_pool.keys())[0]
            sample_col = node.column_pool[sample_key]
            self.logger.info(f"[Column Pool] Sample column {sample_key}:")
            self.logger.info(f"              schedules_x has {len(sample_col.get('schedules_x', {}))} entries")
            if sample_col.get('schedules_x'):
                first_schedule_key = list(sample_col['schedules_x'].keys())[0]
                self.logger.info(
                    f"              First entry: {first_schedule_key} = {sample_col['schedules_x'][first_schedule_key]}")

        # Show distribution
        col_per_profile = {}
        for (p, _) in node.column_pool.keys():
            col_per_profile[p] = col_per_profile.get(p, 0) + 1

        self.logger.info(f"[Column Pool] Distribution across profiles:")
        for p in sorted(col_per_profile.keys())[:5]:
            self.logger.info(f"  Profile {p}: {col_per_profile[p]} columns")
        if len(col_per_profile) > 5:
            self.logger.info(f"  ... and {len(col_per_profile) - 5} more profiles")

    def _finalize_and_print_results(self):
        """
        Finalize the Branch-and-Price solve and print results.

        Updates statistics and prints comprehensive results.
        """
        # Update total time
        self.stats['total_time'] = time.time() - self.start_time

        # Calculate final gap
        self.update_gap()

        # Print detailed results
        self._print_always("\n" + "=" * 100)
        self._print_always(" BRANCH-AND-PRICE RESULTS ".center(100, "="))
        self._print_always("=" * 100)

        # Termination status
        if not self.open_nodes:
            self.logger.info("✅ Status: OPTIMAL (all nodes explored)")
        elif self.gap < 1e-4:
            self.logger.info(f"✅ Status: OPTIMAL (gap < 0.01%)")
        else:
            self.logger.warning(f"⚠️  Status: INCOMPLETE (time/node limit reached)")

        # Bounds and gap
        self.logger.info("Objective Bounds:")
        self.logger.info(f"  Lower Bound (LP): {self.best_lp_bound:.6f}")
        if self.incumbent < float('inf'):
            self.logger.info(f"  Upper Bound (IP): {self.incumbent:.6f}")
            if self.gap < float('inf'):
                self.logger.info(f"  Gap:              {self.gap:.4%}")
            else:
                self.logger.info(f"  Gap:              ∞")
        else:
            self.logger.info(f"  Upper Bound (IP): None found")
            self.logger.info(f"  Gap:              ∞")

        # Node statistics
        self.logger.info("Node Statistics:")
        self.logger.info(f"  Total Nodes:      {self.stats['nodes_explored']}")
        self.logger.info(f"  Nodes Fathomed:   {self.stats['nodes_fathomed']}")
        self.logger.info(f"  Nodes Branched:   {self.stats['nodes_branched']}")
        self.logger.info(f"  Open Nodes:       {len(self.open_nodes)}")

        # Algorithm statistics
        self.logger.info("Algorithm Statistics:")
        self.logger.info(f"  Branching Strategy:   {self.branching_strategy.upper()}")
        self.logger.info(
            f"  Search Strategy:      {'Depth-First (DFS)' if self.search_strategy == 'dfs' else 'Best-Fit (BFS)'}")
        self.logger.info(f"  Total CG Iterations:  {self.stats['total_cg_iterations']}")
        self.logger.info(f"  IP Solves:            {self.stats['ip_solves']}")
        self.logger.info(f"  Incumbent Updates:    {self.stats['incumbent_updates']}")
        self.logger.info(f"  Total Time:           {self.stats['total_time']:.2f}s")

        # Root node information
        self.logger.info("Root Node Information:")
        root = self.nodes[0]
        self.logger.info(f"  Status:           {root.status}")
        self.logger.info(f"  LP Bound:         {root.lp_bound:.6f}")
        self.logger.info(f"  Is Integral:      {root.is_integral}")
        if root.most_fractional_var:
            frac = root.most_fractional_var
            self.logger.info(f"  Most Frac Var:    {frac['var_name']} = {frac['value']:.6f}")

        # Tree structure (if nodes were explored)
        if self.stats['nodes_explored'] > 1:
            self.logger.info("Search Tree Structure:")
            self.logger.info(f"  Max Depth Reached: {max(node.depth for node in self.nodes.values())}")

            # Count nodes by status and fathom reason
            status_counts = {}
            fathom_reasons = {}

            for node in self.nodes.values():
                status_counts[node.status] = status_counts.get(node.status, 0) + 1

                if node.fathom_reason:
                    fathom_reasons[node.fathom_reason] = fathom_reasons.get(node.fathom_reason, 0) + 1

            for status, count in sorted(status_counts.items()):
                self.logger.info(f"  {status.capitalize():20}: {count}")

            if fathom_reasons:
                self.logger.info(f"\n  Fathoming Breakdown:")
                for reason, count in sorted(fathom_reasons.items()):
                    reason_display = reason.replace('_', ' ').title()
                    self.logger.info(f"    {reason_display:25}: {count}")

        # Detailed Processing Log
        if self.stats['node_processing_order']:
            self._print_always("-" * 100)
            self._print_always(" NODE PROCESSING LOG ".center(100, "-"))
            self._print_always("-" * 100)

            order_str = " -> ".join(map(str, self.stats['node_processing_order']))
            self._print_always(f"Processing Order: {order_str}\n")

            if self.search_strategy == 'bfs' and self.stats['bfs_decision_log']:
                self._print_always("BFS Decision Breakdown:")
                for log in self.stats['bfs_decision_log']:
                    self._print_always(f"  Iteration {log['iteration']}:")

                    # Format the list of open nodes for printing
                    open_nodes_str = ", ".join(
                        [f"(Node {nid}, LP {b:.2f})" for b, nid in reversed(log['open_nodes_state'])])
                    self._print_always(f"    - Open Nodes (Ranked): [{open_nodes_str}]")
                    self._print_always(
                        f"    - Decision: Chose Node {log['chosen_node_id']} with the best LP bound of {log['chosen_node_bound']:.4f}.")
            self._print_always("-" * 100)

        # Solution quality
        if self.incumbent < float('inf') and self.incumbent_solution:
            self.logger.info("Best Solution Found:")
            self.logger.info(f"  Objective Value:  {self.incumbent:.6f}")
            self.logger.info(f"  Found at:         Node {self._find_incumbent_node()}")

            # Print some solution details if available
            if 'LOS' in self.incumbent_solution:
                los_values = [v for v in self.incumbent_solution['LOS'].values() if v > 0]
                if los_values:
                    self.logger.info(f"  Avg LOS:          {sum(los_values) / len(los_values):.2f}")
                    self.logger.info(f"  Max LOS:          {max(los_values)}")

        self.logger.info("=" * 100)

    def _find_incumbent_node(self):
        """
        Find which node produced the current incumbent.

        Returns:
            int: Node ID where incumbent was found, or 0 if unknown
        """
        # Search for integral node with matching objective
        for node_id, node in self.nodes.items():
            if node.is_integral and abs(node.lp_bound - self.incumbent) < 1e-5:
                return node_id

        # If not found in nodes, might be from heuristic
        return 0

    def visualize_tree(self, layout='hierarchical', detailed=False, save_path=None):
        """
        Visualize the Branch-and-Price search tree.

        Args:
            layout: 'hierarchical' or 'radial'
            detailed: If True, show detailed information
            save_path: Path to save visualization (optional)
        """
        from tree_visualization import BnPTreeVisualizer

        visualizer = BnPTreeVisualizer(self)

        if detailed:
            visualizer.plot_detailed(save_path=save_path)
        else:
            visualizer.plot(layout=layout, save_path=save_path)

        visualizer.print_tree_statistics()

    def export_tree_graphviz(self, filename='bnp_tree.dot'):
        """Export tree to Graphviz format."""
        from tree_visualization import BnPTreeVisualizer

        visualizer = BnPTreeVisualizer(self)
        visualizer.export_to_graphviz(filename)

    def _run_ip_heuristic(self, current_node_count):
        """
        Periodic IP heuristic: Solve RMP as IP without branching constraints.

        Based on Brunner (2010): Every N nodes, solve the RMP as IP with all
        generated columns but WITHOUT branching constraints. This enlarges the
        feasible region and may find better integer solutions.

        Args:
            current_node_count: Number of nodes explored so far

        Returns:
            bool: True if incumbent was improved
        """
        # Check if we should run the heuristic
        if self.ip_heuristic_frequency <= 0:
            return False

        if current_node_count % self.ip_heuristic_frequency != 0:
            return False

        self.logger.info(f"\n{'─' * 100}")
        self.logger.info(f" IP HEURISTIC (Node {current_node_count}) ".center(100, "─"))
        self.logger.info(f"{'─' * 100}")
        self.logger.info("Solving RMP as IP without branching constraints...")

        master = self.cg_solver.master

        # Save original variable types and bounds
        original_vtypes = {}
        original_bounds = {}

        for var in master.lmbda.values():
            original_vtypes[var.VarName] = var.VType
            original_bounds[var.VarName] = (var.LB, var.UB)

            # Set to integer
            var.VType = gu.GRB.INTEGER

            # CRITICAL: Remove branching bounds to enlarge feasible region
            # This is the key difference from solving at a specific node
            var.LB = 0.0
            var.UB = gu.GRB.INFINITY

        master.Model.update()

        # Solve as IP with time limit
        master.Model.Params.OutputFlag = 0  # Silent
        master.Model.Params.TimeLimit = 60  # 1 minute time limit

        try:
            self.logger.info("  Optimizing...")
            master.Model.optimize()

            improved = False

            if master.Model.status == gu.GRB.OPTIMAL:
                ip_obj = master.Model.objVal

                if ip_obj < self.incumbent - 1e-5:
                    # Found better solution!
                    old_incumbent = self.incumbent
                    self.incumbent = ip_obj
                    self.incumbent_solution = master.finalDicts(
                        self.cg_solver.global_solutions,
                        self.cg_solver.app_data, None
                    )
                    lambda_assignments = {}
                    for (p, a), var in master.lmbda.items():
                        if var.X > 0:
                            lambda_assignments[(p, a)] = int(round(var.X))
                    self.incumbent_lambdas = lambda_assignments
                    self.stats['incumbent_updates'] += 1
                    self.update_gap()

                    self.logger.info(f"\n  ✅ IMPROVED INCUMBENT FOUND!")
                    self.logger.info(f"     Old incumbent: {old_incumbent:.6f}")
                    self.logger.info(f"     New incumbent: {self.incumbent:.6f}")
                    self.logger.info(f"     Improvement:   {old_incumbent - self.incumbent:.6f}")
                    self.logger.info(f"     New gap:       {self.gap:.4%}\n")

                    # Fathom open nodes by bound
                    fathomed_count = self._fathom_by_bound()
                    if fathomed_count > 0:
                        self.logger.info(f"  🔪 Fathomed {fathomed_count} open nodes by bound")

                    improved = True
                else:
                    self.logger.warning(f"  ⚠️  IP solution not improving: {ip_obj:.6f} >= {self.incumbent:.6f}")

            elif master.Model.status == gu.GRB.TIME_LIMIT:
                if master.Model.SolCount > 0:
                    ip_obj = master.Model.objVal
                    self.logger.info(f"  ⏱️  Time limit reached, best solution: {ip_obj:.6f}")

                    if ip_obj < self.incumbent - 1e-5:
                        old_incumbent = self.incumbent
                        self.incumbent = ip_obj
                        self.incumbent_solution = master.finalDicts(
                            self.cg_solver.global_solutions,
                            self.cg_solver.app_data, None
                        )
                        lambda_assignments = {}
                        for (p, a), var in master.lmbda.items():
                            if var.X > 0:
                                lambda_assignments[(p, a)] = int(round(var.X))
                        self.incumbent_lambdas = lambda_assignments
                        self.stats['incumbent_updates'] += 1
                        self.update_gap()

                        self.logger.info(f"     Updated incumbent: {old_incumbent:.6f} → {self.incumbent:.6f}")

                        fathomed_count = self._fathom_by_bound()
                        if fathomed_count > 0:
                            self.logger.info(f"  🔪 Fathomed {fathomed_count} open nodes")

                        improved = True
                else:
                    self.logger.warning(f"  ⚠️  Time limit, no feasible solution found")
            else:
                self.logger.error(f"  ❌ IP solve unsuccessful (status={master.Model.status})")

        except Exception as e:
            self.logger.error(f"  ❌ Error during IP heuristic: {e}")
            improved = False

        finally:
            # Restore original variable types and bounds
            for var in master.lmbda.values():
                var.VType = original_vtypes[var.VarName]
                var.LB = original_bounds[var.VarName][0]
                var.UB = original_bounds[var.VarName][1]

            master.Model.Params.OutputFlag = 0
            master.Model.Params.TimeLimit = gu.GRB.INFINITY
            master.Model.update()

        self.stats['ip_solves'] += 1
        self.logger.info(f"{'─' * 100}\n")

        return improved

    def _fathom_by_bound(self):
        """
        Fathom all open nodes whose LP bound is >= incumbent.

        Returns:
            int: Number of nodes fathomed
        """
        fathomed_count = 0
        nodes_to_remove = []

        # The items in open_nodes depend on the strategy
        if self.search_strategy == 'bfs':
            nodes_to_check = [node_id for _, node_id in self.open_nodes]
        else:  # dfs
            nodes_to_check = self.open_nodes

        for node_id in nodes_to_check:
            node = self.nodes[node_id]

            # Check if node's LP bound is worse than incumbent
            if node.lp_bound >= self.incumbent - 1e-5:
                node.status = 'fathomed'
                node.fathom_reason = 'bound_after_heuristic'
                self.stats['nodes_fathomed'] += 1
                nodes_to_remove.append(node_id)
                fathomed_count += 1

                self.logger.info(f"     Fathomed node {node_id}: LP={node.lp_bound:.6f} >= Inc={self.incumbent:.6f}")

        # Remove from open nodes
        if nodes_to_remove:
            if self.search_strategy == 'bfs':
                self.open_nodes = [(b, i) for b, i in self.open_nodes if i not in nodes_to_remove]
            else:  # dfs
                self.open_nodes = [i for i in self.open_nodes if i not in nodes_to_remove]

        return fathomed_count

    def _get_branching_profile(self, node):
        """
        Extract the branching profile from node's constraints.

        Returns:
            int or None: Profile that was branched on (n), or None if root
        """
        if not node.branching_constraints:
            return None

        # Get the most recent branching constraint (last one added)
        last_constraint = node.branching_constraints[-1]

        # Both SP and MP branching have a 'profile' attribute
        if hasattr(last_constraint, 'profile'):
            return last_constraint.profile

        return None

    # In branch_and_price.py

    def _check_and_fathom_parents(self, node_id):
        """
        Check if parent nodes can be fathomed after a child is fathomed.

        A parent can be fathomed if ALL its children are fathomed.
        This recursively propagates up the tree.

        Args:
            node_id: ID of the node that was just fathomed
        """
        node = self.nodes[node_id]

        if node.parent_id is None:
            return  # Root node has no parent

        parent = self.nodes[node.parent_id]

        # Only check if parent is 'branched' (not already fathomed)
        if parent.status != 'branched':
            return

        # Find all children of the parent
        children = [n for n in self.nodes.values() if n.parent_id == parent.node_id]

        # Check if ALL children are fathomed
        all_children_fathomed = all(child.status == 'fathomed' for child in children)

        if all_children_fathomed:
            # Determine best bound among children for fathom reason
            child_bounds = [child.lp_bound for child in children]
            best_child_bound = min(child_bounds) if child_bounds else float('inf')

            # Fathom parent
            parent.status = 'fathomed'
            parent.fathom_reason = 'all_children_fathomed'
            parent.lp_bound = best_child_bound  # Update with best child bound

            self.logger.info(f"\n  ✅ Parent Node {parent.node_id} fathomed: All children fathomed")
            self.logger.info(f"     Children: {[c.node_id for c in children]}")
            self.logger.info(f"     Best child bound: {best_child_bound:.6f}")

            # Recursively check grandparent
            self._check_and_fathom_parents(parent.node_id)

    # In branch_and_price.py

    def extract_optimal_schedules(self, include_all_patients=True):
        """
        Extract optimal schedules from the incumbent solution.

        Disaggregates profile-level solution to individual patient schedules.
        Based on Paper: "ex-post disaggregation step reconstructs recipient-level
        assignments from the profile-based solution"

        Args:
            include_all_patients: If True, include P_Pre and P_Post patients

        Returns:
            dict: {
                'patient_schedules': {patient_id: schedule_info},
                'objective_value': float,
                'total_los': int,
                'utilization': dict
            }
        """
        if self.incumbent_solution is None:
            self.logger.error("No incumbent solution available!")
            return None

        print("\n" + "=" * 100)
        print(" EXTRACTING OPTIMAL SCHEDULES ".center(100, "="))
        print("=" * 100)

        # Find the node with the incumbent solution
        incumbent_node = self._find_incumbent_node()
        if incumbent_node == 0:
            incumbent_node = self._get_best_integral_node()

        node = self.nodes[incumbent_node]
        lambda_assignments = self.incumbent_lambdas

        print(f"\nExtracting from Node {incumbent_node}")
        print(f"  Objective Value: {self.incumbent:.6f}")
        print(f"  Status: {node.status}")
        print(f"  Assignments: {lambda_assignments}")

        # Disaggregate to individual patients
        patient_schedules = {}
        profile_counters = {}

        for (profile, col_id), count in sorted(lambda_assignments.items()):
            print(f"\n  Profile {profile}, Column {col_id}: {count} patients")

            # Get schedule from column pool
            if (profile, col_id) not in node.column_pool:
                self.logger.warning(f"    ⚠️  Column ({profile},{col_id}) not in pool!")
                continue

            col_data = node.column_pool[(profile, col_id)]

            # Extract schedule information
            schedules_x = col_data.get('schedules_x', {})
            schedules_los = col_data.get('schedules_los', {})

            # Get LOS
            los_value = list(schedules_los.values())[0] if schedules_los else 0

            # Disaggregate: Assign this schedule to 'count' patients
            if profile not in profile_counters:
                profile_counters[profile] = 0

            for i in range(count):
                # Create unique patient ID
                patient_id = f"P{profile}_{profile_counters[profile]}"
                profile_counters[profile] += 1

                # Extract therapist assignment
                assigned_therapist = None
                for (p, j, t, a), val in schedules_x.items():
                    if p == profile and val > 0.5:
                        assigned_therapist = j
                        break

                # Extract daily schedule
                daily_schedule = {}
                for (p, j, t, a), val in schedules_x.items():
                    if p == profile and val > 0.5:
                        if t not in daily_schedule:
                            daily_schedule[t] = []
                        daily_schedule[t].append({
                            'therapist': j,
                            'session': 'human'
                        })

                # Check for AI sessions
                if 'y_data' in col_data:
                    y_data = col_data['y_data']
                    for (p, d, _), val in y_data.items():
                        if p == profile and val > 0.5:
                            if d not in daily_schedule:
                                daily_schedule[d] = []
                            daily_schedule[d].append({
                                'therapist': None,
                                'session': 'AI'
                            })

                # Store patient schedule
                patient_schedules[patient_id] = {
                    'profile': profile,
                    'column': col_id,
                    'therapist': assigned_therapist,
                    'los': los_value,
                    'entry_day': self.cg_solver.Entry_agg[profile],
                    'required_sessions': self.cg_solver.Req_agg[profile],
                    'daily_schedule': daily_schedule,
                    'total_sessions': sum(len(sessions) for sessions in daily_schedule.values())
                }

                self.logger.info(f"    Patient {patient_id}: Therapist {assigned_therapist}, "
                                 f"LOS={los_value}, Sessions={patient_schedules[patient_id]['total_sessions']}")

        # Calculate statistics
        total_los = sum(s['los'] for s in patient_schedules.values())
        avg_los = total_los / len(patient_schedules) if patient_schedules else 0

        # Therapist utilization
        therapist_workload = {}
        for patient_info in patient_schedules.values():
            for day, sessions in patient_info['daily_schedule'].items():
                for session in sessions:
                    if session['therapist'] is not None:
                        t = session['therapist']
                        if t not in therapist_workload:
                            therapist_workload[t] = {}
                        if day not in therapist_workload[t]:
                            therapist_workload[t][day] = 0
                        therapist_workload[t][day] += 1

        # Summary
        self.logger.info("\n" + "=" * 100)
        self.logger.info(" OPTIMAL SOLUTION SUMMARY ".center(100, "="))
        self.logger.info("=" * 100)
        self.logger.info(f"\nPatients:")
        self.logger.info(f"  Total Patients: {len(patient_schedules)}")
        self.logger.info(f"  Total LOS: {total_los}")
        self.logger.info(f"  Average LOS: {avg_los:.2f}")

        self.logger.info(f"\nProfiles:")
        for profile, count in sorted(profile_counters.items()):
            self.logger.info(f"  Profile {profile}: {count} patients")

        self.logger.info(f"\nTherapist Utilization:")
        for t in sorted(therapist_workload.keys()):
            total_sessions = sum(therapist_workload[t].values())
            days_worked = len(therapist_workload[t])
            avg_daily = total_sessions / days_worked if days_worked > 0 else 0
            self.logger.info(f"  Therapist {t}: {total_sessions} sessions over {days_worked} days "
                             f"(avg {avg_daily:.1f}/day)")

        self.logger.info("=" * 100)

        print(f"\nActive columns: {len(lambda_assignments)}")

        print(f"Total expected patients (Nr_agg): {sum(self.cg_solver.Nr_agg[k] for k in sorted(self.cg_solver.P_F + self.cg_solver.P_Post))}")
        print(f"Profiles in lambda_assignments: {set(p for p, _ in lambda_assignments.keys())}")
        print(f"P_Focus: {self.cg_solver.P_F}")
        print(f"P_Post: {self.cg_solver.P_Post}")
        print(f"P_Join: {self.cg_solver.P_Join}")

        return {
            'patient_schedules': patient_schedules,
            'objective_value': self.incumbent,
            'total_los': total_los,
            'avg_los': avg_los,
            'therapist_utilization': therapist_workload,
            'profile_distribution': profile_counters
        }

    def _get_best_integral_node(self):
        """
        Find the best integral node (lowest bound among integral nodes).

        Returns:
            int: Node ID of best integral node
        """
        integral_nodes = [
            (node.lp_bound, node.node_id)
            for node in self.nodes.values()
            if node.is_integral
        ]

        if not integral_nodes:
            return 0  # Return root if no integral node found

        # Return node with lowest bound
        return min(integral_nodes)[1]

    def print_detailed_schedule(self, patient_id, schedule_info):
        """
        Print a detailed schedule for a specific patient.

        Args:
            patient_id: Patient identifier
            schedule_info: Schedule information from extract_optimal_schedules
        """
        print("\n" + "=" * 80)
        print(f" SCHEDULE FOR {patient_id} ".center(80, "="))
        print("=" * 80)

        print(f"\nPatient Information:")
        print(f"  Profile: {schedule_info['profile']}")
        print(f"  Assigned Therapist: {schedule_info['therapist']}")
        print(f"  Entry Day: {schedule_info['entry_day']}")
        print(f"  Length of Stay: {schedule_info['los']} days")
        print(f"  Required Sessions: {schedule_info['required_sessions']}")
        print(f"  Total Sessions: {schedule_info['total_sessions']}")

        print(f"\nDaily Schedule:")
        print(f"  {'Day':<8} {'Therapist':<12} {'Type':<10}")
        print("  " + "-" * 40)

        for day in sorted(schedule_info['daily_schedule'].keys()):
            sessions = schedule_info['daily_schedule'][day]
            for session in sessions:
                therapist = session['therapist'] if session['therapist'] else "N/A"
                session_type = session['session']
                print(f"  {day:<8} {therapist:<12} {session_type:<10}")

        print("=" * 80)

    def export_schedules_to_csv(self, filename='optimal_schedules.csv'):
        """
        Export optimal schedules to CSV file.

        Args:
            filename: Output filename
        """
        import pandas as pd

        schedules = self.extract_optimal_schedules()
        if not schedules:
            return

        # Create rows for CSV
        rows = []
        for patient_id, info in schedules['patient_schedules'].items():
            for day, sessions in info['daily_schedule'].items():
                for session in sessions:
                    rows.append({
                        'patient_id': patient_id,
                        'profile': info['profile'],
                        'day': day,
                        'therapist': session['therapist'] if session['therapist'] else 'AI',
                        'session_type': session['session'],
                        'los': info['los'],
                        'entry_day': info['entry_day']
                    })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

        self.logger.info(f"\n✅ Schedules exported to {filename}")
        self.logger.info(f"   Total rows: {len(df)}")