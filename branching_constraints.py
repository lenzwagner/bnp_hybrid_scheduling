import sys
from abc import ABC, abstractmethod
import gurobipy as gu


class BranchingConstraint(ABC):
    """
    Base class for all branching constraints in Branch-and-Price.

    Each branching constraint must define how it affects:
    1. The master problem (restricting which columns can be used)
    2. The subproblems (restricting which new columns can be generated)
    3. Column compatibility (checking if existing columns satisfy the constraint)
    """

    @abstractmethod
    @abstractmethod
    def apply_to_master(self, master, node):
        """
        Apply this constraint to the master problem.

        Args:
            master: MasterProblem_d instance
            node: BnPNode instance (for accessing column_pool)
        """
        pass

    @abstractmethod
    def apply_to_subproblem(self, subproblem):
        """
        Apply this constraint to a subproblem.

        Args:
            subproblem: Subproblem instance
        """
        pass

    @abstractmethod
    def is_column_compatible(self, column_data):
        """
        Check if a column is compatible with this constraint.

        Args:
            column_data: Dictionary containing column information

        Returns:
            bool: True if column satisfies the constraint
        """
        pass

    @abstractmethod
    def get_dual_contribution(self, for_profile):
        """
        Get the dual variable contribution for pricing.

        Args:
            for_profile: Profile index of the subproblem being priced

        Returns:
            float: Contribution to reduced cost from this constraint's dual
        """
        pass


class SPVariableBranching(BranchingConstraint):
    """
    Branching on Subproblem Variable x_{njt}

    Based on Paper Section 3.2.4, Equations (branch:sub2) and (branch:sub3):
    - Left branch:  x_{njt} = 0  (no assignment of profile n to agent j in period t)
    - Right branch: x_{njt} = 1  (force assignment of profile n to agent j in period t)

    Master Problem Impact (Equation branch:sub2):
    - Left:  sum_{a: chi^a_{njt}=1} Lambda_{na} <= floor(beta)
    - Right: sum_{a: chi^a_{njt}=1} Lambda_{na} >= ceil(beta)

    Subproblem Impact (Equation branch:sub3):
    - Left:  x_{njt} = 0 (fix variable to 0)
    - Right: x_{njt} = 1 (fix variable to 1)

    Attributes:
        profile: Profile index n
        agent: Agent index j (or therapist type if aggregated)
        period: Period index t
        value: 0 for left branch, 1 for right branch
        level: Level in search tree (for dual variable tracking)
        dual_left: Dual variable for left constraint (delta^L_{nl})
        dual_right: Dual variable for right constraint (delta^R_{nl})
    """

    def __init__(self, profile_n, agent_j, period_t, value, dir, level,
                 floor_val, ceil_val):
        self.profile = profile_n
        self.agent = agent_j
        self.period = period_t
        self.value = value
        self.level = level
        self.dir = dir
        self.floor = floor_val
        self.ceil = ceil_val
        self.dual_left = 0.0
        self.dual_right = 0.0
        self.master_constraint = None  # Store reference to created constraint

    def apply_to_master(self, master, node):
        """
        Add constraint to master problem using node's column pool.

        Paper Equation (branch:sub2):
        Left:  sum_{a: chi^a_{njt}=1} Lambda_{na} = 0  (no assignment allowed)
        Right: sum_{a: chi^a_{njt}=1} Lambda_{na} >= 1  (at least one assignment)
        """
        n, j, t = self.profile, self.agent, self.period

        print(f'\n    {"=" * 80}')
        print(f'    [SP Branch] Applying constraint for Node {node.node_id}')
        print(f'    [SP Branch] Profile {n}, Agent {j}, Period {t}')

        # ✅ Find all columns in NODE's column pool where chi^a_{njt} = 1
        relevant_columns = []

        columns_with_id_1 = {key: value for key, value in node.column_pool.items() if key[1] == 1}
        print(columns_with_id_1)

        sys.exit()

        # Filter all columns for profile n
        for (p, a), col_data in node.column_pool.items():
            if p != n:
                continue  # Only check columns for this profile

            schedules_x = col_data.get('schedules_x', {})

            # Check if this column has assignment (n, j, t)
            # Key format in schedules_x: (n, j, t, 0) - always with 0 at the end!
            assignment_key = (n, j, t, 0)

            if assignment_key in schedules_x:
                value = schedules_x[assignment_key]
                print(f'    [SP Branch] Column {a}: schedules_x[{assignment_key}] = {value}')

                if value > 0.5:  # Assignment exists
                    relevant_columns.append(a)
                    print(f'    [SP Branch]   ✅ Column {a} added to relevant_columns')

        print(f'    [SP Branch] Relevant columns: {relevant_columns}')
        print(f'    {"=" * 80}\n')

        print('Gallo', assignment_key, schedules_x, sep="\n")

        if not relevant_columns:
            print(f'    ⚠️  [SP Branch] No relevant columns found - constraint trivially satisfied')
            return

        # Verify Lambda variables exist
        existing_lambdas = [(n, a) for a in relevant_columns if (n, a) in master.lmbda]
        print(f'    [SP Branch] Lambda variables exist: {existing_lambdas}')

        if not existing_lambdas:
            print(f'    ❌ [SP Branch] ERROR: No Lambda variables found!')
            return

        # Create constraint expression
        lhs = gu.quicksum(master.lmbda[n, a] for (n_key, a) in existing_lambdas)
        print('LHS',lhs)

        if self.dir == 'left':  # Left branch
            self.master_constraint = master.Model.addConstr(
                lhs <= self.floor,
                name=f"sp_branch_L{self.level}_{n}_{j}_{t}"
            )
        else:  # Right branch
            self.master_constraint = master.Model.addConstr(
                lhs >= self.ceil,
                name=f"sp_branch_R{self.level}_{n}_{j}_{t}"
            )

        master.Model.update()

        # Verify constraint was added
        for c in master.Model.getConstrs():
            if c.ConstrName == self.master_constraint.ConstrName:
                print(f'    ✅ [SP Branch] Constraint verified in model!')
                break

    def apply_to_subproblem(self, subproblem):
        """
        Fix variable in subproblem.

        Paper Equation (branch:sub3):
        - Left:  x_{njt} = 0
        - Right: x_{njt} = 1
        """
        if subproblem.P != self.profile:
            return  # Not relevant for this subproblem

        # Key format in subproblem: x[p, j, t, iteration]
        var_key = (self.profile, self.agent, self.period, subproblem.itr)

        if var_key in subproblem.x:
            # Fix variable to branching value
            if self.dir == 'left':
                subproblem.x[var_key].LB = 0
                subproblem.x[var_key].UB = 0
            else:
                subproblem.x[var_key].LB = 1
                subproblem.x[var_key].UB = 1
            subproblem.Model.update()

    def is_column_compatible(self, column_data):
        """
        Check if column's assignment chi^a_{njt} matches the constraint.

        Args:
            column_data: Dict with keys 'index', 'schedules_x', etc.

        Returns:
            bool: True if compatible
        """
        if column_data.get('index') != self.profile:
            return True  # Not relevant for other profiles

        # Check the assignment in schedules_x
        # schedules_x format: {(p, j, t, a): value}
        schedule = column_data.get('schedules_x', {})

        # Look for assignment (profile, agent, period, *)
        for (p, j, t, a), val in schedule.items():
            if p == self.profile and j == self.agent and t == self.period:
                # Found the assignment, check if it matches constraint
                return (val == self.value)

        # If not found in schedule, it's implicitly 0
        return (self.value == 0)

    def get_dual_contribution(self, for_profile):
        """
        Get dual contribution for pricing objective.

        IMPORTANT: This dual is only relevant for the specific profile n
        that this constraint applies to. Other profiles are unaffected.

        Paper Equation (branch:sub4):
        For SP_n: Reduced cost includes: - sum_{l in L(n)} (delta^L_{nl} + delta^R_{nl})
        For SP_m (m != n): No contribution

        Args:
            for_profile: Profile index of the subproblem being priced

        Returns:
            float: -(delta^L + delta^R) if for_profile == self.profile, else 0.0
        """
        # Only apply to the specific profile this constraint was created for
        if for_profile != self.profile:
            return 0.0

        if self.master_constraint is None:
            return 0.0

        # Get dual value from master constraint
        try:
            dual_val = self.master_constraint.Pi
            return -dual_val  # Negative because it's subtracted in reduced cost
        except:
            return 0.0

    def __repr__(self):
        return (f"SPBranch(profile={self.profile}, agent={self.agent}, "
                f"period={self.period}, dir={self.dir}, level={self.level})")


class MPVariableBranching(BranchingConstraint):
    """
    Branching on Master Problem Variable Lambda_{na}

    Based on Paper Section 3.2.4, Equation (branch_mp1):
    - Left branch:  Lambda_{na} <= floor(Lambda_hat)
    - Right branch: Lambda_{na} >= ceil(Lambda_hat)

    Master Problem Impact:
    - Simply set variable bounds on Lambda_{na}

    Subproblem Impact:
    - Left branch: Add no-good cut to prevent regenerating column a
    - Right branch: No modification needed (column becomes more attractive)

    Attributes:
        profile: Profile index n
        column: Column index a
        bound: floor(Lambda) or ceil(Lambda)
        direction: 'left' or 'right'
        original_schedule: Schedule of the forbidden column (for no-good cut)
    """

    def __init__(self, profile_n, column_a, bound, direction, original_schedule=None):
        self.profile = profile_n
        self.column = column_a
        self.bound = bound  # floor or ceil of Lambda
        self.direction = direction  # 'left' or 'right'
        self.original_schedule = original_schedule  # For no-good cut

    def apply_to_master(self, master, node):  # ← node Parameter hinzugefügt (auch wenn hier nicht gebraucht)
        """
        Set variable bounds on Lambda_{na}.
        """
        var = master.lmbda.get((self.profile, self.column))

        if var is None:
            print(f"    ❌ ERROR: Variable Lambda[{self.profile},{self.column}] not found in master!")
            return

        # Set bounds
        if self.direction == 'left':
            master.set_branching_bound(var, 'ub', self.bound)
        else:  # right
            master.set_branching_bound(var, 'lb', self.bound)

        master.Model.update()

    def apply_to_subproblem(self, subproblem):
        """
        Left branch: Add no-good cut (Equation no_good_cut_disagg)
        Right branch: No modification needed

        No-good cut prevents regenerating the exact same column:
        sum_{(j,t): chi^a_{njt}=1} (1-x_{njt}) + sum_{(j,t): chi^a_{njt}=0} x_{njt} >= 1
        """
        if subproblem.P != self.profile:
            return  # Not relevant for other profiles

        if self.direction == 'right':
            return  # No SP modification needed on right branch

        # Left branch: Add no-good cut
        if self.original_schedule is None or len(self.original_schedule) == 0:
            print(f"      ⚠️ WARNING: No original_schedule for no-good cut!")
            print(f"         Column ({self.profile}, {self.column}) can be regenerated!")
            return

        print(f"      [No-Good Cut] Adding for profile {self.profile}, column {self.column}")
        print(f"                    Schedule has {len(self.original_schedule)} assignments")

        # Build no-good constraint
        terms = []
        schedule_pattern = []  # For debugging

        for (p, j, t, a_orig), chi_value in self.original_schedule.items():
            if p != self.profile:
                continue

            var_key = (p, j, t, subproblem.itr)
            if var_key not in subproblem.x:
                continue

            x_var = subproblem.x[var_key]

            if chi_value == 1:
                # Position where chi was 1 must be different
                terms.append(1 - x_var)
                schedule_pattern.append(f"x[{j},{t}]=1")
            else:
                # Position where chi was 0 must be different
                terms.append(x_var)
                schedule_pattern.append(f"x[{j},{t}]=0")

        if terms:
            subproblem.Model.addConstr(
                gu.quicksum(terms) >= 1,
                name=f"no_good_p{self.profile}_c{self.column}"
            )
            print(f"                    Pattern: {', '.join(schedule_pattern[:5])}...")
            print(f"                    Added constraint with {len(terms)} terms")
        else:
            print(f"      ⚠️ WARNING: No terms in no-good cut (empty schedule?)")

    def is_column_compatible(self, column_data):
        """
        Check if column is compatible with this constraint.

        For MP Variable Branching:
        - ALL columns remain in the model!
        - We only set variable bounds, we don't filter columns
        - The no-good cut in the subproblem prevents regeneration

        Args:
            column_data: Dict with column information

        Returns:
            bool: Always True for MP branching (columns are not filtered)
        """
        # MP branching doesn't filter columns - only sets variable bounds
        return True

    def get_dual_contribution(self, for_profile):
        """
        Get dual contribution for pricing objective.

        MP variable branching only affects existing columns via bounds.
        These bounds are handled in the dual space naturally by Gurobi,
        so no explicit contribution to reduced cost is needed.

        Args:
            for_profile: Profile index (not used for MP branching)

        Returns:
            float: 0.0
        """
        return 0.0

    def __repr__(self):
        return (f"MPBranch(profile={self.profile}, column={self.column}, "
                f"{self.direction}, bound={self.bound})")