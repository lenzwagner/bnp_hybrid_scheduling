import sys
import time
from bnp_node import BnPNode
import gurobipy as gu



class BranchAndPrice:
    """
    Branch-and-Price Algorithm - Minimal version for Phase 1.

    This version creates only the root node and solves it via Column Generation.
    This is functionally identical to the current CG implementation,
    but in the B&P structure.

    In later phases we will add:
    - Branching logic
    - Multiple nodes
    - Fathoming
    - Search strategy

    Attributes:
        nodes: Dictionary of all nodes {node_id -> BnPNode}
        node_counter: Counter for unique node IDs
        open_nodes: List of open nodes (later: queue for DFS)
        incumbent: Best found IP solution (upper bound)
        best_lp_bound: Best LP bound of all nodes (lower bound)
        gap: Optimality gap
        cg_solver: Reference to Column Generation solver
    """

    def __init__(self, cg_solver, branching_strategy='mp'):
        """
        Initialize Branch-and-Price with existing CG solver.

        Args:
            cg_solver: ColumnGeneration object (already initialized with setup())
            branching_strategy: 'mp' for MP variable branching, 'sp' for SP variable branching
        """
        # Node management
        self.nodes = {}  # {node_id -> BnPNode}
        self.node_counter = 0
        self.open_nodes = []  # List of open node IDs

        # Global bounds
        self.incumbent = float('inf')  # Best IP solution (upper bound)
        self.incumbent_solution = None
        self.best_lp_bound = float('inf')  # Best LP bound (lower bound)
        self.gap = float('inf')

        # Reference to CG solver
        self.cg_solver = cg_solver

        # Start solutions
        self.start_x = self.cg_solver.start_x
        self.start_los = self.cg_solver.start_los

        # Branching Configuration
        self.branching_strategy = branching_strategy  # 'mp' or 'sp'

        # Statistics
        self.stats = {
            'nodes_explored': 0,
            'nodes_fathomed': 0,
            'nodes_branched': 0,
            'total_cg_iterations': 0,
            'total_time': 0,
            'incumbent_updates': 0,
            'ip_solves': 0
        }

        # Timing
        self.start_time = None

        print("\n" + "=" * 100)
        print(" BRANCH-AND-PRICE INITIALIZED ".center(100, "="))
        print("=" * 100)
        print(f"CG Solver ready with {len(self.cg_solver.P_Join)} patients")
        print(f"Branching strategy: {self.branching_strategy.upper()}")
        print("=" * 100 + "\n")


        # Initialize LP-folder
        import os
        os.makedirs("LPs/MP/LPs", exist_ok=True)
        os.makedirs("LPs/MP/SOLs", exist_ok=True)

        os.makedirs("LPs/SPs/left", exist_ok=True)
        os.makedirs("LPs/SPs/right", exist_ok=True)


    def _initialize_col_id_counters(self):
        """
        Initialize global column ID counters from initial CG solution.

        Scans all existing columns in cg_solver.global_solutions
        and sets next_col_id to max(existing) + 1 for each profile.
        """
        print("\n[Init] Initializing global column ID counters...")

        for profile in self.cg_solver.P_Join:
            # Find all column IDs for this profile
            profile_col_ids = [
                col_id for (p, col_id) in self.cg_solver.global_solutions.get('x', {}).keys()
                if p == profile
            ]

            if profile_col_ids:
                next_id = max(profile_col_ids) + 1
            else:
                next_id = 1

            self.next_col_id_per_profile[profile] = next_id
            print(f"  Profile {profile}: next_col_id = {next_id} (found {len(profile_col_ids)} existing columns)")

        print(f"[Init] Column ID counters initialized for {len(self.next_col_id_per_profile)} profiles\n")

    def get_next_col_id(self, profile):
        """
        Get the next available global column ID for a profile.

        This ensures column IDs are unique across the entire B&P tree.

        Args:
            profile: Profile index

        Returns:
            int: Next available column ID
        """
        if profile not in self.next_col_id_per_profile:
            self.next_col_id_per_profile[profile] = 1

        col_id = self.next_col_id_per_profile[profile]
        self.next_col_id_per_profile[profile] += 1

        return col_id

    def create_root_node(self):
        """
        Create root node with initial columns from CG heuristic.

        Returns:
            BnPNode: The root node (ID=0, depth=0)
        """
        node = BnPNode(node_id=0, depth=0)

        # Transfer initial columns from CG solver
        for (p, old_col_id) in self.cg_solver.global_solutions.get('x', {}).keys():
            # WICHTIG: Verwende die RICHTIGE column_id aus dem CG solver
            col_id = old_col_id

            # Extract schedules_x from global_solutions
            x_solution = self.cg_solver.global_solutions['x'][(p, old_col_id)]

            # Remap keys: (p, agent, period, old_iteration) -> (p, agent, period, col_id)
            schedules_x = {}
            for (p_key, agent, period, old_iter), value in x_solution.items():
                # Use col_id instead of old_iter
                schedules_x[(p_key, agent, period, col_id)] = value

            # Extract schedules_los
            schedules_los = {}
            if (p, old_col_id) in self.cg_solver.global_solutions.get('LOS', {}):
                los_solution = self.cg_solver.global_solutions['LOS'][(p, old_col_id)]
                # Remap keys: (p, old_iteration) -> (p, col_id)
                for (p_key, old_iter), value in los_solution.items():
                    schedules_los[(p_key, col_id)] = value

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

        # Store node
        self.nodes[0] = node
        self.open_nodes.append(0)

        print(f"\n{'=' * 100}")
        print(" ROOT NODE CREATED ".center(100, "="))
        print(f"{'=' * 100}")
        print(f"Root node initialized with {len(node.column_pool)} initial columns")
        print(f"Columns distribution:")

        # Show column distribution by profiles
        col_per_profile = {}
        for (p, _) in node.column_pool.keys():
            col_per_profile[p] = col_per_profile.get(p, 0) + 1

        for p in sorted(col_per_profile.keys())[:5]:
            print(f"  Profile {p}: {col_per_profile[p]} columns")
        if len(col_per_profile) > 5:
            print(f"  ... and {len(col_per_profile) - 5} more profiles")

        # Debug: Show sample column structure
        if node.column_pool:
            sample_key = list(node.column_pool.keys())[0]
            sample_col = node.column_pool[sample_key]
            print(f"\n  Sample column {sample_key}:")
            print(f"    schedules_x: {len(sample_col.get('schedules_x', {}))} entries")
            print(f"    schedules_los: {len(sample_col.get('schedules_los', {}))} entries")
            print(f"    x_list: {len(sample_col.get('x_list', []))} values")
            print(f"    los_list: {len(sample_col.get('los_list', []))} values")

            # Show first key format
            if sample_col.get('schedules_x'):
                first_key = list(sample_col['schedules_x'].keys())[0]
                print(f"    First schedules_x key: {first_key}")

        print(f"{'=' * 100}\n")

        return node

    def solve_root_node(self):
        """
        Solve root node via Column Generation.

        After CG converges, solve master as IP to get initial incumbent.

        Returns:
            tuple: (lp_bound, is_integral, most_frac_info)
        """
        print("\n" + "=" * 100)
        print(" SOLVING ROOT NODE ".center(100, "="))
        print("=" * 100 + "\n")

        # REMOVED: callback for incumbent computation during CG
        # We compute it AFTER CG converges instead

        # Solve with Column Generation
        self.cg_solver.solve_cg()

        # After CG converges: compute initial incumbent by solving RMP as IP
        print("\n" + "=" * 100)
        print(" COMPUTING INITIAL INCUMBENT ".center(100, "="))
        print("=" * 100)
        print("Column Generation converged. All columns generated.")
        print("Solving Root Master Problem as IP to get initial upper bound...\n")

        self._compute_root_incumbent()

        # After incumbent computation: Get final LP results
        print("\n[Root] Final LP relaxation check...")
        self.cg_solver.master.solRelModel()
        is_integral, lp_bound, most_frac_info = self.cg_solver.master.check_fractionality()

        # Save final Root Node LP
        self.cg_solver.master.Model.write('Final_Root.lp')
        print(f"\n[Root] ✅ Saved final root node LP to: Final_Root.lp")

        # Update root node
        root_node = self.nodes[0]
        root_node.lp_bound = lp_bound
        root_node.is_integral = is_integral
        root_node.most_fractional_var = most_frac_info

        # Update root node's column pool with ALL generated columns
        self._update_node_column_pool(root_node)

        # Update node status
        if is_integral:
            root_node.status = 'integral'
            root_node.fathom_reason = 'integral'
            print(f"\n✅ ROOT NODE IS INTEGRAL (LP)!")
        else:
            root_node.status = 'open'
            print(f"\n⚠️  ROOT NODE IS FRACTIONAL (LP)")

        print(f"   LP Bound: {lp_bound:.6f}")
        print(f"   Incumbent: {self.incumbent:.6f}" if self.incumbent < float('inf') else "   Incumbent: None")

        # Update global bounds
        self.best_lp_bound = lp_bound
        self.update_gap()

        # Save initial Root-Sol
        self.cg_solver.master.Model.write('LPs/MP/SOLs/Root.sol')

        print(f"\n{'=' * 100}")
        print(" ROOT NODE SOLVED ".center(100, "="))
        print(f"{'=' * 100}\n")

        return lp_bound, is_integral, most_frac_info

    def should_fathom(self, node):
        """
        Determine if a node should be fathomed.

        A node is fathomed if:
        1. Its LP solution is integral (we found an IP solution)
        2. Its LP bound >= incumbent (no better solution possible)
        3. Its LP is infeasible

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
                    self.cg_solver.app_data
                )
                self.stats['incumbent_updates'] += 1
                self.update_gap()

                print(f"   New gap: {self.gap:.4%}\n")

            # TODO: Fix fathoming logic

            return True

        # Check 2: Bound worse than incumbent
        if node.lp_bound >= self.incumbent - 1e-5:
            node.status = 'fathomed'
            node.fathom_reason = 'bound'
            print(f"   Node {node.node_id} fathomed by bound: "
                  f"LP={node.lp_bound:.6f} >= UB={self.incumbent:.6f}")
            return True

        # Check 3: Infeasible
        if node.lp_bound == float('inf'):
            node.status = 'fathomed'
            node.fathom_reason = 'infeasible'
            print(f"   Node {node.node_id} fathomed: LP infeasible")
            return True

        # Node cannot be fathomed
        return False

    def _compute_root_incumbent(self):
        """
        Compute initial incumbent by solving root RMP as IP after CG converges.

        This provides the initial upper bound for Branch-and-Price.
        Based on Paper Section 3.2.3: "Initialization & Feasibility Heuristic"

        The RMP is solved as IP with all columns generated during root node CG.
        This is more effective than solving after the first iteration because
        all improving columns have been found.

        Returns:
            tuple: (success: bool, incumbent_value: float)
        """
        print("=" * 100)
        print("Solving Root Master Problem as Integer Program...")
        print("This provides the initial upper bound (incumbent) for B&P.")
        print("=" * 100 + "\n")

        self.stats['ip_solves'] += 1
        master = self.cg_solver.master

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

            print("[IP Solve] Starting optimization...")
            master.Model.optimize()

            # Check solution status
            if master.Model.status == gu.GRB.OPTIMAL:
                ip_obj = master.Model.objVal

                # Update incumbent
                if ip_obj < self.incumbent:
                    self.incumbent = ip_obj + 10
                    self.incumbent_solution = master.finalDicts(
                        self.cg_solver.global_solutions,
                        self.cg_solver.app_data
                    )
                    self.stats['incumbent_updates'] += 1
                    self.update_gap()

                    print(f"\n{'=' * 100}")
                    print("✅ INITIAL INCUMBENT FOUND ".center(100, "="))
                    print(f"{'=' * 100}")
                    print(f"IP Objective:     {self.incumbent:.6f}")
                    print(f"LP Bound (root):  {master.Model.objBound:.6f}" if hasattr(master.Model, 'objBound') else "")
                    print(f"Gap:              {self.gap:.4%}")
                    print(f"{'=' * 100}\n")

                    success = True
                    result_obj = ip_obj
                else:
                    print(f"\n⚠️  IP solution not better than current incumbent")
                    print(f"   IP Objective:      {ip_obj:.6f}")
                    print(f"   Current Incumbent: {self.incumbent:.6f}\n")
                    success = False
                    result_obj = ip_obj

            elif master.Model.status == gu.GRB.TIME_LIMIT:
                print(f"\n⚠️  IP solve hit time limit")
                if master.Model.SolCount > 0:
                    ip_obj = master.Model.objVal
                    print(f"   Best found solution: {ip_obj:.6f}")
                    if ip_obj < self.incumbent:
                        self.incumbent = ip_obj
                        self.incumbent_solution = master.finalDicts(
                            self.cg_solver.global_solutions,
                            self.cg_solver.app_data
                        )
                        self.stats['incumbent_updates'] += 1
                        self.update_gap()
                        print(f"   Updated incumbent: {self.incumbent:.6f}\n")
                        success = True
                        result_obj = ip_obj
                    else:
                        success = False
                        result_obj = ip_obj
                else:
                    print(f"   No feasible solution found within time limit\n")
                    success = False
                    result_obj = float('inf')

            else:
                print(f"❌ IP solve unsuccessful (status={master.Model.status})")
                success = False
                result_obj = float('inf')

            # Restore continuous relaxation for future LP solves
            for var in master.lmbda.values():
                var.VType = original_vtypes[var.VarName]

            master.Model.Params.OutputFlag = 0
            master.Model.Params.TimeLimit = float('inf')
            master.Model.update()

            return success, result_obj

        except Exception as e:
            print(f"❌ Error during IP solve: {e}\n")

            # Restore original variable types
            for var in master.lmbda.values():
                if var.VarName in original_vtypes:
                    var.VType = original_vtypes[var.VarName]

            master.Model.Params.OutputFlag = 0
            master.Model.update()

            return False, float('inf')

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

        print("\n" + "=" * 100)
        print(" BRANCH-AND-PRICE SOLVE ".center(100, "="))
        print("=" * 100)
        print(f"Time limit: {time_limit}s")
        print(f"Max nodes: {max_nodes}")
        print(f"Branching strategy: {self.branching_strategy.upper()}")
        print("=" * 100 + "\n")

        # ========================================
        # PHASE 1: CREATE AND SOLVE ROOT NODE
        # ========================================
        root_node = self.create_root_node()
        lp_bound, is_integral, frac_info = self.solve_root_node()
        self.stats['nodes_explored'] = 1

        # Check if root can be fathomed
        if self.should_fathom(root_node):
            print(f"✅ Root node fathomed: {root_node.fathom_reason}")
            print(f"   Solution is optimal!\n")
            self.stats['nodes_fathomed'] = 1
            self.open_nodes.remove(0)
            self._finalize_and_print_results()
            return self._get_results_dict()

        # Root needs branching
        print(f"\n{'=' * 100}")
        print(" ROOT NODE REQUIRES BRANCHING ".center(100, "="))
        print(f"{'=' * 100}\n")

        # Branch on root
        branching_type, branching_info = self.select_branching_candidate(root_node)

        if not branching_type:
            print(f"⚠️  Could not find branching candidate despite fractional solution!")
            self._finalize_and_print_results()
            return self._get_results_dict()

        # Create child nodes
        if branching_type == 'mp':
            left_child, right_child = self.branch_on_mp_variable(root_node, branching_info)
        else:  # 'sp'
            left_child, right_child = self.branch_on_sp_variable(root_node, branching_info)

        # Mark root as branched
        root_node.status = 'branched'
        self.stats['nodes_branched'] += 1
        self.open_nodes.remove(0)  # Remove root from open nodes

        # ========================================
        # PHASE 2: MAIN BRANCH-AND-PRICE LOOP
        # ========================================
        print(f"\n{'=' * 100}")
        print(" MAIN BRANCH-AND-PRICE LOOP ".center(100, "="))
        print(f"{'=' * 100}\n")

        iteration = 0

        while self.open_nodes and iteration < max_nodes:
            iteration += 1

            # Check time limit
            elapsed = time.time() - self.start_time
            if elapsed > time_limit:
                print(f"\n⏱️  Time limit reached: {elapsed:.2f}s > {time_limit}s")
                break

            # ========================================
            # SELECT NEXT NODE (DFS: LIFO)
            # ========================================
            current_node_id = self.open_nodes.pop()
            print(f"   🔎 Open nodes stack: {self.open_nodes}")
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
                lp_bound, is_integral, most_frac_info = self.solve_node_with_cg(
                    current_node, max_cg_iterations=50
                )
            except Exception as e:
                print(f"❌ Error solving node {current_node_id}: {e}")
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
            if self.should_fathom(current_node):
                print(f"✅ Node {current_node_id} fathomed: {current_node.fathom_reason}")
                self.stats['nodes_fathomed'] += 1

                # Print current status
                print(f"\n   Status after fathoming:")
                print(f"   ├─ Best LB: {self.best_lp_bound:.6f}")
                print(f"   ├─ Incumbent: {self.incumbent:.6f}" if self.incumbent < float(
                    'inf') else "   ├─ Incumbent: None")
                print(f"   ├─ Gap: {self.gap:.4%}" if self.gap < float('inf') else "   ├─ Gap: ∞")
                print(f"   └─ Open nodes: {len(self.open_nodes)}\n")

                continue


            # ========================================
            # NODE NOT FATHOMED → BRANCH
            # ========================================
            print(f"\n⚠️  Node {current_node_id} requires branching (LP is fractional)")

            # Select branching candidate
            branching_type, branching_info = self.select_branching_candidate(current_node)

            if not branching_type:
                print(f"❌ Could not find branching candidate at node {current_node_id}")
                print(f"   Marking as fathomed (should not happen!)")
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

            print(f"\n✅ Created child nodes:")
            print(f"   ├─ Left:  Node {left_child.node_id} (path: '{left_child.path}')")
            print(f"   └─ Right: Node {right_child.node_id} (path: '{right_child.path}')")
            print(f"\n   Open nodes queue: {self.open_nodes}")

        # ========================================
        # FINALIZATION
        # ========================================
        print(f"\n{'=' * 100}")
        print(" BRANCH-AND-PRICE TERMINATED ".center(100, "="))
        print(f"{'=' * 100}")

        # Determine termination reason
        if not self.open_nodes:
            print(f"✅ All nodes explored - Tree complete!")
        elif iteration >= max_nodes:
            print(f"⚠️  Node limit reached: {iteration} >= {max_nodes}")
            print(f"   {len(self.open_nodes)} nodes remain open")
        else:
            print(f"⏱️  Time limit reached")
            print(f"   {len(self.open_nodes)} nodes remain open")

        self._finalize_and_print_results()
        return self._get_results_dict()


    def _print_final_results(self):
        """Print final results."""
        print("\n" + "=" * 100)
        print(" BRANCH-AND-PRICE RESULTS ".center(100, "="))
        print("=" * 100)
        print(f"Status: Phase 1 Complete (Root Node Only)")
        print(f"")
        print(f"Bounds:")
        print(f"  LP Bound (LB):  {self.best_lp_bound:.6f}")
        print(f"  Incumbent (UB): {self.incumbent:.6f}" if self.incumbent < float('inf') else "  Incumbent (UB): None")
        print(f"  Gap:            {self.gap:.4%}" if self.gap < float('inf') else "  Gap:            ∞")
        print(f"")
        print(f"Statistics:")
        print(f"  Nodes Explored:   {self.stats['nodes_explored']}")
        print(f"  Nodes Fathomed:   {self.stats['nodes_fathomed']}")
        print(f"  CG Iterations:    {self.stats['total_cg_iterations']}")
        print(f"  IP Solves:        {self.stats['ip_solves']}")
        print(f"  Incumbent Updates: {self.stats['incumbent_updates']}")
        print(f"  Total Time:       {self.stats['total_time']:.2f}s")
        print(f"")
        print(f"Root Node Info:")
        root = self.nodes[0]
        print(f"  Status:         {root.status}")
        print(f"  Is Integral:    {root.is_integral}")
        print(f"  LP Bound:       {root.lp_bound:.6f}")
        if root.most_fractional_var:
            frac = root.most_fractional_var
            print(f"  Most Frac Var:  {frac['var_name']} = {frac['value']:.6f} (dist={frac['fractionality']:.6f})")
        if root.fathom_reason:
            print(f"  Fathom Reason:  {root.fathom_reason}")
        print("=" * 100 + "\n")

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

    def _root_node_callback(self, iteration, cg_solver):
        """
        Callback executed after each CG iteration at root node.

        After the first iteration, we solve the master as IP to get
        an initial incumbent (upper bound).

        Args:
            iteration: Current CG iteration number
            cg_solver: Reference to ColumnGeneration instance
        """
        # Only compute incumbent after first iteration
        if iteration == 1:
            print(f"\n{'─' * 100}")
            print(" COMPUTING INITIAL INCUMBENT (after CG iteration 1) ".center(100, "─"))
            print(f"{'─' * 100}")

            self.stats['ip_solves'] += 1

            try:
                # Solve current master as IP
                print("[Callback] Solving master problem as IP...")

                # Set all lambda variables to integer
                for var in cg_solver.master.lmbda.values():
                    var.VType = gu.GRB.INTEGER

                cg_solver.master.Model.Params.OutputFlag = 0  # Silent
                cg_solver.master.Model.optimize()

                if cg_solver.master.Model.status == 2:  # GRB.OPTIMAL
                    ip_obj = cg_solver.master.Model.objVal

                    # Update incumbent if better
                    if ip_obj < self.incumbent:
                        self.incumbent = ip_obj + 10
                        self.incumbent_solution = cg_solver.master.finalDicts(
                            cg_solver.global_solutions,
                            cg_solver.app_data
                        )
                        self.stats['incumbent_updates'] += 1
                        self.update_gap()

                        print(f"[Callback] ✅ Initial incumbent found: {ip_obj:.6f}")
                        print(f"[Callback]    Current LP bound: {cg_solver.master.Model.objVal:.6f}")
                        print(f"[Callback]    Gap: {self.gap:.4%}")
                    else:
                        print(f"[Callback] IP solution not better: {ip_obj:.6f} >= {self.incumbent:.6f}")
                else:
                    print(f"[Callback] ⚠️  IP solve unsuccessful (status={cg_solver.master.Model.status})")

                # Reset variables to continuous for remaining CG iterations
                for var in cg_solver.master.lmbda.values():
                    var.VType = gu.GRB.CONTINUOUS  # GRB.CONTINUOUS

                cg_solver.master.Model.Params.OutputFlag = 1  # Verbose again

            except Exception as e:
                print(f"[Callback] ❌ Error computing initial incumbent: {e}")

            print(f"{'─' * 100}\n")

    # ============================================================================
    # BRANCHING LOGIC
    # ============================================================================

    def select_branching_candidate(self, node):
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
            return self._select_sp_branching_candidate(node)
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
            print(f"⚠️  Unknown variable type: {var_name}")
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

    def _select_sp_branching_candidate(self, node):
        """
        Select most fractional beta_{njt} for SP branching.

        beta_{njt} = sum_{a: chi^a_{njt}=1} Lambda_{na}

        Returns:
            tuple: ('sp', branching_info) or (None, None)
        """
        # Compute beta values for all (n,j,t) combinations
        beta_values = {}

        # Access master solution to compute beta
        master = self.cg_solver.master

        for (n, a), var in master.lmbda.items():
            lambda_val = var.X

            if lambda_val < 1e-6:
                continue

            # Get chi values for this column from all_schedules
            # Format: all_schedules[(p, j, t, a)] = chi_value
            for (p, j, t, a_col), chi_val in master.all_schedules.items():
                if p == n and a_col == a and chi_val > 0.5:
                    key = (n, j, t)
                    beta_values[key] = beta_values.get(key, 0.0) + lambda_val

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
                # Check if this is better (higher fractionality, or tie-break by n,j,t)
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

        if best_candidate is None:
            return None, None

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

        print(f"\n{'=' * 100}")
        print(f" BRANCHING ON MP VARIABLE ".center(100, "="))
        print(f"{'=' * 100}")
        print(f"Branching on Lambda[{n},{a}] = {lambda_value:.6f}")
        print(f"  Left:  Lambda[{n},{a}] <= {floor_val}")
        print(f"  Right: Lambda[{n},{a}] >= {ceil_val}")


        # Get original schedule for no-good cut
        original_schedule = None
        if (n, a) in parent_node.column_pool:
            original_schedule = parent_node.column_pool[(n, a)].get('schedules_x', {})
            print(f"\n  ✅ Found column ({n},{a}) in parent's column pool")
            print(f"     Schedule has {len(original_schedule)} assignments")

            # Show first few assignments
            if original_schedule:
                sample_assignments = list(original_schedule.items())[:3]
                for key, val in sample_assignments:
                    print(f"       {key}: {val}")
        else:
            print(f"\n  ❌ ERROR: Column ({n},{a}) NOT found in parent's column pool!")
            print(
                f"     Available columns for profile {n}: {[col_id for (p, col_id) in parent_node.column_pool.keys() if p == n]}")
            print(f"     No-good cut cannot be added!")

        # -------------------------
        # LEFT CHILD
        # -------------------------
        self.node_counter += 1
        left_child = BnPNode(
            node_id=self.node_counter,
            parent_id=parent_node.node_id,
            depth=parent_node.depth + 1,
            path = parent_node.path + 'l'
        )

        #if left_child.depth == 3:
            #print(f"\n🛑 DEBUG STOPPER: Reached depth {left_child.depth}")
            #print(f"Path: {left_child.path}")
            #sys.exit()

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
        self._save_subproblem_for_branching_profile(left_child, n, True)

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
        self._save_subproblem_for_branching_profile(right_child, n, False)


        # -------------------------
        # STORE NODES
        # -------------------------
        self.nodes[left_child.node_id] = left_child
        self.nodes[right_child.node_id] = right_child

        # Add to open nodes (DFS: right first, then left, so left is processed first)
        self.open_nodes.append(right_child.node_id)
        self.open_nodes.append(left_child.node_id)

        # Update parent status
        parent_node.status = 'branched'

        print(f"  Created left child:  Node {left_child.node_id} (depth {left_child.depth})")
        print(f"  Created right child: Node {right_child.node_id} (depth {right_child.depth})")
        print(f"{'=' * 100}\n")

        self.stats['nodes_branched'] += 1

        return left_child, right_child

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

        print(f"    Column inheritance: {inherited_count} inherited, {filtered_count} filtered")

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
        print(f"\n{'─' * 100}")
        print(f" SOLVING NODE {node.node_id} (path: '{node.path}', depth {node.depth}) ".center(100, "─"))
        print(f"{'─' * 100}")
        print(f"Branching constraints: {len(node.branching_constraints)}")
        print(f"Column pool size: {len(node.column_pool)}")

        # Show column distribution
        cols_per_profile = {}
        for (p, _) in node.column_pool.keys():
            cols_per_profile[p] = cols_per_profile.get(p, 0) + 1
        print(f"Columns per profile (sample): {dict(list(cols_per_profile.items())[:3])}")
        print(f"{'─' * 100}\n")

        # 1. Build master problem and save LP for this node
        master = self._build_master_for_node(node)
        master.Model.update()
        master.Model.write(f"LPs/MP/LPs/mp_root_{node.node_id}.lp")
        print(f"    [Master] Saved LP to: LPs/MP/LPs/mp_root_{node.node_id}.lp")

        # 2. Column Generation loop
        threshold = self.cg_solver.threshold  # Use same threshold as CG
        cg_iteration = 0

        # Node time limit
        node_start_time = time.time()
        NODE_TIME_LIMIT = 300

        while cg_iteration < max_cg_iterations:
            if time.time() - node_start_time > NODE_TIME_LIMIT:
                print(f"⏱️  Node {node.node_id} time limit reached")
                break

            cg_iteration += 1

            print(f"    [CG Iter {cg_iteration}] Solving master LP...")

            # Solve master as LP
            if node.node_id == 18:
                master.Model.write(f"LP_{cg_iteration}_{node.node_id}.lp")
            master.solRelModel()
            if node.node_id == 18:
                master.Model.write(f"_{cg_iteration}_{node.node_id}.lp")

            if master.Model.status != 2:  # GRB.OPTIMAL
                print(f"    ⚠️  Master infeasible or unbounded at node {node.node_id}")
                return float('inf'), False, None

            current_lp_obj = master.Model.objVal
            print(f"    [CG Iter {cg_iteration}] LP objective: {current_lp_obj:.6f}")

            # Get duals from master
            duals_td, duals_p = master.getDuals()

            print(self.branching_strategy)

            # Get branching constraint duals if SP-branching is used
            branching_duals = {}
            if self.branching_strategy == 'sp':
                branching_duals = self._get_branching_constraint_duals(master, node)

            # 3. Solve subproblems for all profiles
            new_columns_found = False
            counter = 0
            columns_added_this_iter = 0

            for profile in self.cg_solver.P_Join:
                # Build and solve subproblem with branching constraints
                sp = self._build_subproblem_for_node(
                    profile, node, duals_td, duals_p, branching_duals
                )
                sp.solModel()

                # Check reduced cost
                if sp.Model.status == 2 and sp.Model.objVal < -threshold:
                    print(f'Red. cost for profile {profile} : {sp.Model.objVal}')

                    # Add column to node and master
                    self._add_column_from_subproblem(sp, profile, node, master)
                    new_columns_found = True
                    columns_added_this_iter += 1
                    master.Model.update()

            print(f"    [CG Iter {cg_iteration}] Added {columns_added_this_iter} new columns")

            # Check convergence
            if not new_columns_found:
                print(f"    [CG] Converged after {cg_iteration} iterations - no improving columns found")
                break
            master.Model.update()

        # 4. Final LP solve and integrality check
        print(f"\n    [Node {node.node_id}] Final LP solve...")
        master.Model.write(f"LPs/MP/LPs/mp_final_{node.node_id}.lp")
        master.solRelModel()
        master.Model.write(f"LPs/MP/SOLs/mp_node_{node.node_id}.sol")
        is_integral, lp_obj, most_frac_info = master.check_fractionality()

        # >>> NEU: Wenn integral → .sol speichern und beenden <<<
        if is_integral:
            print(f"\n✅ INTEGRAL SOLUTION FOUND AT NODE {node.node_id}!")
            print(f"   LP Bound: {lp_obj:.6f}")
            #sys.exit(0)

        # Store results in node
        node.lp_bound = lp_obj
        node.is_integral = is_integral
        node.most_fractional_var = most_frac_info

        print(f"\n    [Node {node.node_id}] Results:")
        print(f"      LP Bound: {lp_obj:.6f}")
        print(f"      Is Integral: {is_integral}")
        print(f"      CG Iterations: {cg_iteration}")
        print(f"      Final column pool: {len(node.column_pool)} columns")

        if most_frac_info:
            print(f"      Most fractional: {most_frac_info['var_name']} = {most_frac_info['value']:.6f}")

        print(f"{'─' * 100}\n")

        self.stats['total_cg_iterations'] += cg_iteration

        return lp_obj, is_integral, most_frac_info

    def _build_master_for_node(self, node):
        """
        Build master problem for a node with inherited columns and branching constraints.
        """
        from masterproblem import MasterProblem_d

        print(f"    [Master] Building master problem for node {node.node_id}...")

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

        print(f"    [Master] Basic model built with {len(master.Model.getConstrs())} constraints")

        sp_branching_active = False

        # Load columns
        print(f"    [Master] Loading {len(node.column_pool)} columns from pool...")


        for (profile, col_id), col_data in node.column_pool.items():

            if col_id >= 2:

                # Add schedules to master
                schedules_x = col_data.get('schedules_x', {})
                schedules_los = col_data.get('schedules_los', {})

                if not schedules_x:
                    print(f"      ⚠️ WARNING: Column ({profile},{col_id}) has empty schedules_x!")
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
                    col_coefs = col_coefs + branching_coefs
                    print(f"      [Column {profile},{col_id}] Added {len(branching_coefs)} branching coefficients")

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
            print(f"    [Master] Applying {len(node.branching_constraints)} branching constraints...")

            for constraint in node.branching_constraints:
                print('Cons', constraint)

                constraint.apply_to_master(master)
                # Check if this is SP branching (adds constraints)
                if hasattr(constraint, 'master_constraint') and constraint.master_constraint is not None:
                    sp_branching_active = True

            master.Model.update()
            print(f"    [Master] Now have {len(master.Model.getConstrs())} constraints")
            print(f"    [Master] SP-Branching constraints added: {sp_branching_active}")

        print(f"    [Master] Master problem ready:")
        print(f"             - {len(master.lmbda)} lambda variables")
        print(f"             - {len(master.Model.getConstrs())} constraints")

        return master

    def _compute_branching_coefficients_for_column(self, col_data, profile, col_id, branching_constraints):
        """
        Compute coefficients for branching constraints for an existing column.

        For each branching constraint, determine what coefficient this column should have.

        Args:
            col_data: Column data dict with schedules_x, etc.
            profile: Profile index
            col_id: Column ID
            branching_constraints: List of BranchingConstraint objects

        Returns:
            list: Coefficients for each branching constraint (in order they appear in model)
        """
        from branching_constraints import SPVariableBranching, MPVariableBranching

        coefs = []
        schedules_x = col_data.get('schedules_x', {})

        for constraint in branching_constraints:
            if isinstance(constraint, SPVariableBranching):
                # Coefficient is 1 if chi^a_{njt} = 1, else 0
                # Check if this column assigns profile to (agent, period)
                n, j, t = constraint.profile, constraint.agent, constraint.period

                # Look through schedules_x for this assignment
                # Format: {(p, j, t, a): value}
                chi_value = 0
                for (p, j_sched, t_sched, a), val in schedules_x.items():
                    if p == profile and j_sched == j and t_sched == t and a == col_id:
                        chi_value = val
                        break

                coef = 1 if chi_value > 0.5 else 0
                coefs.append(coef)

            elif isinstance(constraint, MPVariableBranching):
                # MP branching uses variable bounds, not linear constraints
                # So coefficient is always 0 (or constraint doesn't exist yet)
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

        These duals must be included in the subproblem pricing objective
        according to Paper Equation (branch:sub4).

        Args:
            master: MasterProblem_d instance
            node: BnPNode

        Returns:
            dict: {(profile, agent, period, level): dual_value}
        """
        branching_duals = {}

        for constraint in node.branching_constraints:
            if hasattr(constraint, 'master_constraint') and constraint.master_constraint:
                try:
                    dual_val = constraint.master_constraint.Pi

                    # Store dual for this branching constraint
                    key = (constraint.profile, constraint.agent, constraint.period, constraint.level)
                    branching_duals[key] = dual_val

                    print(f"      [Duals] SP branching constraint at level {constraint.level}: "
                          f"x[{constraint.profile},{constraint.agent},{constraint.period}]={constraint.value}, "
                          f"dual={dual_val:.6f}")
                except:
                    pass  # Constraint not binding or no dual available

        return branching_duals

    def _build_subproblem_for_node(self, profile, node, duals_td, duals_p, branching_duals=None):
        """
        Build subproblem for a profile at a node with branching constraints.

        Uses node-local column IDs and REAL duals from master LP.

        Args:
            profile: Profile index
            node: BnPNode
            duals_td: Dual variables for capacity constraints
            duals_p: Dual variables for profile constraints
            branching_duals: Dict of branching constraint duals (for SP-branching)

        Returns:
            Subproblem: Subproblem with constraints
        """
        from subproblem import Subproblem

        if branching_duals is None:
            branching_duals = {}

        # Bestimme nächste col_id basierend auf column_pool dieses Nodes
        profile_columns = [col_id for (p, col_id) in node.column_pool.keys() if p == profile]

        if profile_columns:
            next_col_id = max(profile_columns) + 1
        else:
            next_col_id = 1

        # Create subproblem mit echten Duals
        sp = Subproblem(
            self.cg_solver.data,
            duals_p,
            duals_td,
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

        # Modify objective for SP branching constraint duals (Paper Eq. branch:sub4)
        if self.branching_strategy == 'sp' and branching_duals:
            self._add_branching_duals_to_objective(sp, profile, branching_duals)

        sp.Model.update()

        return sp

    def _add_branching_duals_to_objective(self, sp, profile, branching_duals):
        """
        Modify subproblem objective to include branching constraint duals.

        Paper Equation (branch:sub4):
        Objective becomes: F_n - sum π_jt * x_njt - sum (δ^L_nl + δ^R_nl) - γ_n

        Args:
            sp: Subproblem instance
            profile: Profile index
            branching_duals: Dict of branching constraint duals
        """
        # Sum all branching constraint duals for this profile
        dual_sum = 0.0

        for (p, j, t, level), dual_val in branching_duals.items():
            if p == profile:
                dual_sum += dual_val

        if abs(dual_sum) > 1e-10:
            # Modify objective: subtract the sum of branching duals
            current_obj = sp.Model.getObjective()
            sp.Model.setObjective(current_obj - dual_sum, sense=gu.GRB.MINIMIZE)

            print(f"        [SP Objective] Added branching dual contribution: {-dual_sum:.6f}")

    def _add_column_from_subproblem(self, subproblem, profile, node, master):
        """
        Add a column generated from a subproblem to node and master.

        Args:
            subproblem: Solved Subproblem instance
            profile: Profile index
            node: BnPNode
            master: MasterProblem_d instance
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

        master.addLambdaVar(
            profile, col_id,
            lambda_list + x_list,
            los_list
        )

        print(f"        [Column] Added column ({profile}, {col_id}) with reduced cost {subproblem.Model.objVal:.6f}")

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

        print(f"\n{'=' * 100}")
        print(f" BRANCHING ON SP VARIABLE ".center(100, "="))
        print(f"{'=' * 100}")
        print(f"Branching on x[{n},{j},{t}], beta = {beta_val:.6f}")
        print(f"  Left:  x[{n},{j},{t}] = 0")
        print(f"  Right: x[{n},{j},{t}] = 1")

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
            value=0,
            level=left_child.depth
        )

        left_child.branching_constraints = parent_node.branching_constraints.copy()
        left_child.branching_constraints.append(left_constraint)

        self._inherit_columns_from_parent(left_child, parent_node)
        self._save_subproblem_for_branching_profile(left_child, n, True)


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
            value=1,
            level=right_child.depth
        )

        right_child.branching_constraints = parent_node.branching_constraints.copy()
        right_child.branching_constraints.append(right_constraint)

        self._inherit_columns_from_parent(right_child, parent_node)
        self._save_subproblem_for_branching_profile(right_child, n, False)


        # -------------------------
        # STORE NODES
        # -------------------------
        self.nodes[left_child.node_id] = left_child
        self.nodes[right_child.node_id] = right_child

        self.open_nodes.append(right_child.node_id)
        self.open_nodes.append(left_child.node_id)

        parent_node.status = 'branched'

        print(f"  Created left child:  Node {left_child.node_id} (depth {left_child.depth})")
        print(f"  Created right child: Node {right_child.node_id} (depth {right_child.depth})")
        print(f"{'=' * 100}\n")

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
        print(f"\n[Column Pool] Updating node {node.node_id} with generated columns...")

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

        print(f"[Column Pool] Updated: {initial_count} → {final_count} columns (+{added_count} new)")

        # Debug: Show some schedules_x info
        if added_count > 0:
            sample_key = list(node.column_pool.keys())[0]
            sample_col = node.column_pool[sample_key]
            print(f"[Column Pool] Sample column {sample_key}:")
            print(f"              schedules_x has {len(sample_col.get('schedules_x', {}))} entries")
            if sample_col.get('schedules_x'):
                first_schedule_key = list(sample_col['schedules_x'].keys())[0]
                print(
                    f"              First entry: {first_schedule_key} = {sample_col['schedules_x'][first_schedule_key]}")

        # Show distribution
        col_per_profile = {}
        for (p, _) in node.column_pool.keys():
            col_per_profile[p] = col_per_profile.get(p, 0) + 1

        print(f"[Column Pool] Distribution across profiles:")
        for p in sorted(col_per_profile.keys())[:5]:
            print(f"  Profile {p}: {col_per_profile[p]} columns")
        if len(col_per_profile) > 5:
            print(f"  ... and {len(col_per_profile) - 5} more profiles")
        print()

    def _save_subproblem_for_branching_profile(self, node, profile, isLeft = True):
        """
        Build and save the subproblem for a given profile at a node (without solving it).
        Used to debug or inspect the SP used for branching.
        """
        # Dummy duals – since we only want the model structure, not pricing
        dummy_duals_td = {(t, d): 0.0 for t in list(range(1, max(key[1] for key in self.start_x.keys()) + 1)) for d in list(range(1, max(key[2] for key in self.start_x.keys()) + 1))}
        dummy_duals_p = {p: 0.0 for p in self.cg_solver.P_Join}

        # Determine next col_id (not critical for model structure)
        profile_columns = [col_id for (p, col_id) in node.column_pool.keys() if p == profile]
        next_col_id = max(profile_columns) + 1 if profile_columns else 1

        from subproblem import Subproblem
        sp = Subproblem(
            self.cg_solver.data,
            dummy_duals_p,
            dummy_duals_td,
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

        # Apply branching constraints
        for constraint in node.branching_constraints:
            constraint.apply_to_subproblem(sp)

        sp.Model.update()
        if isLeft:
            filename = f"LPs/SPs/left/sp_node_{node.node_id}_profile_{profile}_left.lp"
        else:
            filename = f"LPs/SPs/right/sp_node_{node.node_id}_profile_{profile}_right.lp"
        sp.Model.write(filename)
        print(f"    [Subproblem] Saved SP for profile {profile} to: {filename}")

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
        print("\n" + "=" * 100)
        print(" BRANCH-AND-PRICE RESULTS ".center(100, "="))
        print("=" * 100)

        # Termination status
        if not self.open_nodes:
            print("✅ Status: OPTIMAL (all nodes explored)")
        elif self.gap < 1e-4:
            print(f"✅ Status: OPTIMAL (gap < 0.01%)")
        else:
            print(f"⚠️  Status: INCOMPLETE (time/node limit reached)")

        print()

        # Bounds and gap
        print("Objective Bounds:")
        print(f"  Lower Bound (LP): {self.best_lp_bound:.6f}")
        if self.incumbent < float('inf'):
            print(f"  Upper Bound (IP): {self.incumbent:.6f}")
            if self.gap < float('inf'):
                print(f"  Gap:              {self.gap:.4%}")
            else:
                print(f"  Gap:              ∞")
        else:
            print(f"  Upper Bound (IP): None found")
            print(f"  Gap:              ∞")

        print()

        # Node statistics
        print("Node Statistics:")
        print(f"  Total Nodes:      {self.stats['nodes_explored']}")
        print(f"  Nodes Fathomed:   {self.stats['nodes_fathomed']}")
        print(f"  Nodes Branched:   {self.stats['nodes_branched']}")
        print(f"  Open Nodes:       {len(self.open_nodes)}")

        print()

        # Algorithm statistics
        print("Algorithm Statistics:")
        print(f"  Branching Strategy:   {self.branching_strategy.upper()}")
        print(f"  Total CG Iterations:  {self.stats['total_cg_iterations']}")
        print(f"  IP Solves:            {self.stats['ip_solves']}")
        print(f"  Incumbent Updates:    {self.stats['incumbent_updates']}")
        print(f"  Total Time:           {self.stats['total_time']:.2f}s")

        print()

        # Root node information
        print("Root Node Information:")
        root = self.nodes[0]
        print(f"  Status:           {root.status}")
        print(f"  LP Bound:         {root.lp_bound:.6f}")
        print(f"  Is Integral:      {root.is_integral}")
        if root.most_fractional_var:
            frac = root.most_fractional_var
            print(f"  Most Frac Var:    {frac['var_name']} = {frac['value']:.6f}")

        print()

        # Tree structure (if nodes were explored)
        if self.stats['nodes_explored'] > 1:
            print("Search Tree Structure:")
            print(f"  Max Depth Reached: {max(node.depth for node in self.nodes.values())}")

            # Count nodes by status
            status_counts = {}
            for node in self.nodes.values():
                status_counts[node.status] = status_counts.get(node.status, 0) + 1

            for status, count in sorted(status_counts.items()):
                print(f"  {status.capitalize():15}: {count}")

        print()

        # Solution quality
        if self.incumbent < float('inf') and self.incumbent_solution:
            print("Best Solution Found:")
            print(f"  Objective Value:  {self.incumbent:.6f}")
            print(f"  Found at:         Node {self._find_incumbent_node()}")

            # Print some solution details if available
            if 'LOS' in self.incumbent_solution:
                los_values = [v for v in self.incumbent_solution['LOS'].values() if v > 0]
                if los_values:
                    print(f"  Avg LOS:          {sum(los_values) / len(los_values):.2f}")
                    print(f"  Max LOS:          {max(los_values)}")

        print("=" * 100)
        print()

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