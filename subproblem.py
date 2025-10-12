import gurobipy as gu
import math
from Utils.Generell.utils import *


class Subproblem:
    def __init__(self, df, duals_p, duals_td, p, col_id, Req, Entry, app_data, W_coeff, E_dict, S_Bound,
                 learn_method,
                 reduction=False, num_tangents=10, node_path=''):
        self.reduction = reduction
        self.P = p
        self.W_coeff = W_coeff
        self.Req = Req
        self.app_data = app_data
        self.E = E_dict
        self.node_path = node_path
        self.col_id = col_id
        self.itr = col_id
        self.Entry = Entry
        self.learn_method = learn_method
        self.num_tangents = num_tangents
        self.P_Full = df['P_Full'].dropna().astype(int).unique().tolist()
        self.D_raw = df['D_Ext'].dropna().astype(int).unique().tolist()

        self._init_day_horizon()
        self._init_patient_sets(df)

        self.T = df['T'].dropna().astype(int).unique().tolist()
        self.duals_p = duals_p
        self.duals_td = duals_td
        self.Model = gu.Model("Subproblem")
        self.M = max(self.D) + 1
        self.S_Bound = S_Bound[self.P]
        self.R = list(range(1, 1 + self.S_Bound))
        print(f'Duals for {self.P} in itr. {self.itr}: {self.duals_p,self.duals_td}')

    def _init_day_horizon(self):
        """Initialize the day horizon with optional reduction."""
        if self.reduction:
            self.D_p = self.D_raw
            start_day = max(1, self.Entry[self.P])
            max_los = self.Entry[self.P] + math.ceil(self.Req[self.P] / self.app_data["MS_min"][0]) * \
                      self.app_data["MS"][0] + 2
            end_day = min(max(self.D_raw), max_los)
            self.D = [d for d in self.D_p if start_day <= d <= end_day]
            node_info = f"in Node '{self.node_path}'" if self.node_path else "at root"
            boxed_print(
                f"Reduced subproblem for Patient {self.P} {node_info}: "
                f"Horizon from day {start_day} till day {end_day}"
            )
        else:
            self.D_p = self.D_raw
            self.D = self.D_raw

    def _init_patient_sets(self, df):
        """Initialize patient category sets."""
        self.P_Pre = df['P_Pre'].dropna().astype(int).unique().tolist()
        self.P_Join = df['P_Join'].dropna().astype(int).unique().tolist()
        self.P_Focus = df['P_F'].dropna().astype(int).unique().tolist()
        self.P_Post = df['P_Post'].dropna().astype(int).unique().tolist()

    def compute_tangent_approximation(self, k_learn, theta_base):
        """
        Calculates tangents for the outer approximation of the exponential learning curve.

        f(Y) = theta_base + (1 - theta_base) * (1 - exp(-k * Y))
        f'(Y) = (1 - theta_base) * k_learn * exp(-k_learn * Y)

        Tangent at Y^(s): θ ≤ m_s * Y + β_s
        where: m_s = f'(Y^(s))
               β_s = f(Y^(s)) - m_s * Y^(s)
        """
        support_points = self._get_support_points()
        tangents = []

        for Y_s in support_points:
            f_Y_s = theta_base + (1 - theta_base) * (1 - math.exp(-k_learn * Y_s))
            m_s = (1 - theta_base) * k_learn * math.exp(-k_learn * Y_s)
            beta_s = f_Y_s - m_s * Y_s
            tangents.append((m_s, beta_s, Y_s))

        return tangents

    def compute_tangent_approximation_sigmoid(self, k_learn, infl_point, theta_base):
        """
        Calculates tangents for the outer approximation of the sigmoid learning curve.

        f(Y) = theta_base + (1 - theta_base) / (1 + exp(-k_learn * (Y - infl_point)))
        f'(Y) = (1 - theta_base) * k_learn * exp(-k_learn * (Y - infl_point)) / (1 + exp(-k_learn * (Y - infl_point)))^2
        """
        support_points = self._get_support_points()
        tangents = []

        for Y_s in support_points:
            exp_term = math.exp(-k_learn * (Y_s - infl_point))
            f_Y_s = theta_base + (1 - theta_base) / (1 + exp_term)
            m_s = (1 - theta_base) * k_learn * exp_term / ((1 + exp_term) ** 2)
            beta_s = f_Y_s - m_s * Y_s
            tangents.append((m_s, beta_s, Y_s))

        return tangents

    def _get_support_points(self):
        """Generate support points for tangent approximation."""
        Y_bound = self.S_Bound
        if Y_bound == 0:
            return [0]
        return [i * Y_bound / (self.num_tangents - 1) for i in range(self.num_tangents)]

    def buildModel(self):
        self.genVars()
        self.genCons()
        self.genLearnCons()
        self.genObj()
        self.Model.update()

    def genVars(self):
        """Generate all decision variables."""
        self.l = self.Model.addVars([self.P], self.D, vtype=gu.GRB.BINARY, name="l")
        self.LOS = self.Model.addVars([self.P], [self.col_id], vtype=gu.GRB.INTEGER, name="LOS")
        self.x = self.Model.addVars([self.P], self.T, self.D, [self.col_id], vtype=gu.GRB.BINARY, name="x")
        self.y = self.Model.addVars([self.P], self.D, vtype=gu.GRB.BINARY, name="y")
        self.z = self.Model.addVars([self.P], self.T, vtype=gu.GRB.BINARY, name="z")
        self.w = self.Model.addVars([self.P], self.D, vtype=gu.GRB.BINARY, name="w")
        self.S = self.Model.addVars([self.P], self.D, vtype=gu.GRB.INTEGER, lb=0, name="S")

    def genCons(self):
        """Generate main constraints."""
        p = self.P

        # LOS and discharge constraints
        self._add_los_constraints(p)
        self._add_discharge_constraints(p)

        # Therapist assignment constraints
        self._add_therapist_constraints(p)

        # Daily constraints
        self._add_daily_constraints(p)

        # Minimum sessions constraints
        self._add_minimum_sessions_constraints(p)

        self.Model.update()

    def _add_los_constraints(self, p):
        """Add Length of Stay (LOS) constraints."""
        max_day = max(self.D)
        sum_l = gu.quicksum(self.l[p, d] for d in self.D)

        # LOS bounds when patient is discharged
        self.Model.addLConstr(
            self.LOS[p, self.col_id] >= max_day - self.Entry[p] + 1 - self.M * sum_l,
            name=f'LOS_lower_bound_{p}'
        )
        self.Model.addLConstr(
            self.LOS[p, self.col_id] <= max_day - self.Entry[p] + 1 + self.M * sum_l,
            name=f'LOS_upper_bound_{p}'
        )

        # LOS based on actual discharge day
        discharge_day = gu.quicksum(self.l[p, d] * d for d in self.D)
        self.Model.addLConstr(
            self.LOS[p, self.col_id] >= discharge_day - self.Entry[p] + 1 - self.M * (1 - sum_l),
            name=f'LOS_discharge_lower_{p}'
        )
        self.Model.addLConstr(
            self.LOS[p, self.col_id] <= discharge_day - self.Entry[p] + 1 + self.M * (1 - sum_l),
            name=f'LOS_discharge_upper_{p}'
        )

    def _add_discharge_constraints(self, p):
        """Add discharge-related constraints."""
        if p in self.P_Focus:
            self.Model.addLConstr(
                gu.quicksum(self.l[p, d] for d in self.D) == 1,
                name=f'discharge_required_{p}'
            )
        else:
            self.Model.addLConstr(
                gu.quicksum(self.l[p, d] for d in self.D) <= 1,
                name=f'discharge_optional_{p}'
            )

    def _add_therapist_constraints(self, p):
        """Add therapist assignment constraints."""
        # Single therapist per patient
        self.Model.addLConstr(
            gu.quicksum(self.z[p, t] for t in self.T) == 1,
            name=f'single_therapist_{p}'
        )

        # First day assignment
        self.Model.addLConstr(
            gu.quicksum(self.x[p, t, self.Entry[p], self.col_id] for t in self.T) == 1,
            name=f'first_day_assignment_{p}'
        )

    def _add_daily_constraints(self, p):
        """Add daily eligibility and assignment constraints."""
        max_day = max(self.D)
        entry_day = self.Entry[p]

        # Last day indicator
        for d in range(entry_day + 1, max_day + 1):
            self.Model.addLConstr(
                self.w[p, d] == 1 - gu.quicksum(self.l[p, k] for k in range(entry_day, d)),
                name=f'last_day_indicator_{p}_{d}'
            )

        for d in self.D:
            # Eligibility constraints
            if d < entry_day:
                self.Model.addLConstr(self.w[p, d] == 0, name=f'not_eligible_{p}_{d}')
            elif d == entry_day:
                self.Model.addLConstr(self.w[p, d] == 1, name=f'eligible_first_day_{p}_{d}')

            # Discharge requires therapist assignment
            self.Model.addLConstr(
                self.l[p, d] <= gu.quicksum(self.x[p, t, d, self.col_id] for t in self.T),
                name=f'discharge_needs_therapist_{p}_{d}'
            )

            # Daily assignment
            self.Model.addLConstr(
                self.w[p, d] == gu.quicksum(self.x[p, t, d, self.col_id] for t in self.T) + self.y[p, d],
                name=f'daily_assignment_{p}_{d}'
            )

            # Therapist consistency
            for t in self.T:
                self.Model.addLConstr(
                    self.x[p, t, d, self.col_id] <= self.z[p, t],
                    name=f'therapist_consistency_{p}_{t}_{d}'
                )

    def _add_minimum_sessions_constraints(self, p):
        """Add minimum sessions per milestone constraints."""
        max_day = max(self.D)
        ms_period = self.app_data["MS"][0]
        ms_min = self.app_data["MS_min"][0]

        for d in self.D:
            if d < max_day - ms_period + 1:
                sessions_in_window = gu.quicksum(
                    self.x[p, t, j, self.col_id]
                    for t in self.T
                    for j in range(d, d + ms_period)
                )
                active_days_in_window = gu.quicksum(
                    self.w[p, j] for j in range(d, d + ms_period)
                )

                self.Model.addConstr(
                    sessions_in_window >= ms_min - ms_min * (ms_period - active_days_in_window),
                    name=f'min_sessions_{p}_{d}'
                )

    def genLearnCons(self):
        """Generate learning effect constraints based on method."""
        learn_type = self.app_data["learn_type"][0]

        if learn_type in ['exp', 'sigmoid']:
            if self.learn_method == 'approx':
                self._add_tangent_approximation_constraints()
            elif self.learn_method == 'PWL':
                self._add_pwl_constraints()
            else:
                self._add_linearization_constraints()
        elif learn_type == 'lin':
            self._add_linear_learning_constraints()
        else:
            self._add_basic_learning_constraints()

        self.Model.update()

    def _add_tangent_approximation_constraints(self):
        """Add tangent approximation constraints for exponential/sigmoid learning."""
        learn_type = self.app_data["learn_type"][0]
        k_learn = self.app_data["k_learn"][0]
        theta_base = self.app_data["theta_base"][0]

        print(f'Using Approx: {learn_type}')

        self.App = self.Model.addVars([self.P], self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="App")

        # Compute tangents
        if learn_type == 'exp':
            tangents = self.compute_tangent_approximation(k_learn, theta_base)
            boxed_print(f"Using exponential tangent approximation with {len(tangents)} support points")
        else:  # sigmoid
            infl_point = self.app_data["infl_point"][0]
            tangents = self.compute_tangent_approximation_sigmoid(k_learn, infl_point, theta_base)
            boxed_print(f"Using sigmoid tangent approximation with {len(tangents)} support points")

        p = self.P
        for d in self.D:
            # Define cumulative skipped sessions
            self.Model.addLConstr(
                self.S[p, d] == gu.quicksum(self.y[p, t] for t in range(self.Entry[p], d + 1)),
                name=f"cumulative_skipped_{p}_{d}"
            )

            # Add tangent constraints
            for s, (m_s, beta_s, Y_s) in enumerate(tangents):
                self.Model.addLConstr(
                    self.App[p, d] <= m_s * self.S[p, d] + beta_s,
                    name=f"tangent_{p}_{d}_{s}"
                )

            self.Model.addLConstr(self.App[p, d] <= 1, name=f"App_upper_bound_{p}_{d}")

            # Requirement fulfillment
            self._add_requirement_constraint(p, d)

    def _add_pwl_constraints(self):
        """Add piecewise linear constraints for exponential/sigmoid learning."""
        learn_type = self.app_data["learn_type"][0]
        k = self.app_data["k_learn"][0]
        theta_base = self.app_data["theta_base"][0]

        print(f'Using PWL: {learn_type}')

        self.h_eff = self.Model.addVars([self.P], self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="h_eff")
        self.App = self.Model.addVars([self.P], self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="App")
        self.S = self.Model.addVars([self.P], self.D, vtype=gu.GRB.CONTINUOUS, lb=0, name="S")

        # Prepare PWL breakpoints
        x_values = sorted(list(self.R))

        if learn_type == 'exp':
            y_values = [theta_base + (1 - theta_base) * (1 - math.exp(-k * r)) for r in x_values]
        else:  # sigmoid
            x0 = self.app_data["infl_point"][0]
            y_values = [theta_base + (1 - theta_base) / (1 + math.exp(-k * (r - x0))) for r in x_values]

        p = self.P
        for d in self.D:
            # Define cumulative skipped sessions
            self.Model.addConstr(
                self.S[p, d] == gu.quicksum(self.y[p, t] for t in range(self.Entry[p], d + 1)),
                name=f"cumulative_skipped_{p}_{d}"
            )

            # PWL approximation
            self.Model.addGenConstrPWL(
                self.S[p, d], self.App[p, d], x_values, y_values,
                name=f"pwl_App_{p}_{d}"
            )

            # Effective learning (product of y and App)
            self._add_bilinear_product_constraints(p, d)

            # Requirement fulfillment
            self._add_requirement_constraint(p, d, use_heff=True)

    def _add_linearization_constraints(self):
        """Add exact linearization constraints for exponential/sigmoid learning."""
        learn_type = self.app_data["learn_type"][0]
        k = self.app_data["k_learn"][0]
        theta_base = self.app_data["theta_base"][0]
        infl_point = self.app_data["infl_point"][0]

        print(f'Using Linearization: {learn_type}')

        self.z_pdr = self.Model.addVars(
            [self.P], self.D, list(range(0, self.S_Bound + 1)),
            vtype=gu.GRB.BINARY, name="z_pdr"
        )
        self.h_eff = self.Model.addVars([self.P], self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="h_eff")
        self.App = self.Model.addVars([self.P], self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="App")

        # Precompute learning values
        r_range = list(range(0, self.S_Bound + 1))
        if learn_type == 'exp':
            learning_values = {
                r: theta_base + (1 - theta_base) * (1 - math.exp(-k * r))
                for r in r_range
            }
        else:  # sigmoid
            learning_values = {
                r: theta_base + (1 - theta_base) / (1 + math.exp(-k * (r - infl_point)))
                for r in r_range
            }

        p = self.P
        for d in self.D:
            # SOS1: exactly one r value is selected
            self.Model.addLConstr(
                gu.quicksum(self.z_pdr[p, d, r] for r in r_range) == 1,
                name=f"sos1_{p}_{d}"
            )

            # Define cumulative skipped sessions
            self.Model.addLConstr(
                gu.quicksum(self.y[p, t] for t in range(self.Entry[p], d + 1)) ==
                gu.quicksum(r * self.z_pdr[p, d, r] for r in r_range),
                name=f"cumulative_skipped_{p}_{d}"
            )

            # Define learning effectiveness
            self.Model.addLConstr(
                self.App[p, d] == gu.quicksum(learning_values[r] * self.z_pdr[p, d, r] for r in r_range),
                name=f"learning_effect_{p}_{d}"
            )

            # Effective learning (product of y and App)
            self._add_bilinear_product_constraints(p, d)

            # Requirement fulfillment
            self._add_requirement_constraint_with_heff(p, d)

    def _add_linear_learning_constraints(self):
        """Add linear learning constraints."""
        self.App = self.Model.addVars([self.P], self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="App")

        theta_base = self.app_data["theta_base"][0]
        lin_increase = self.app_data["lin_increase"][0]

        p = self.P
        for d in self.D:
            # Define cumulative skipped sessions
            self.Model.addLConstr(
                self.S[p, d] == gu.quicksum(self.y[p, t] for t in range(self.Entry[p], d + 1)),
                name=f"cumulative_skipped_{p}_{d}"
            )

            # Linear learning effect
            self.Model.addLConstr(
                self.App[p, d] <= theta_base + lin_increase * self.S[p, d],
                name=f"linear_learning_{p}_{d}"
            )
            self.Model.addLConstr(self.App[p, d] <= 1, name=f"App_upper_bound_{p}_{d}")

            # Requirement fulfillment
            self._add_requirement_constraint(p, d)

    def _add_basic_learning_constraints(self):
        """Add basic learning constraints (constant learning factor)."""
        p = self.P
        learn_factor = self.app_data["learn_type"][0]

        for d in self.D:
            sessions = gu.quicksum(
                gu.quicksum(self.x[p, t, j, self.col_id] for j in range(self.D[0], d + 1))
                for t in self.T
            )
            skipped_benefit = gu.quicksum(
                self.y[p, j] * learn_factor for j in range(self.Entry[p], d + 1)
            )

            self.Model.addConstr(
                self.l[p, d] * self.Req[p] <= sessions + skipped_benefit,
                name=f"requirement_{p}_{d}"
            )

    def _add_bilinear_product_constraints(self, p, d):
        """Add constraints for h_eff = y * App (McCormick envelope)."""
        self.Model.addLConstr(
            self.h_eff[p, d] <= self.y[p, d],
            name=f"h_eff_upper_y_{p}_{d}"
        )
        self.Model.addLConstr(
            self.h_eff[p, d] <= self.App[p, d],
            name=f"h_eff_upper_app_{p}_{d}"
        )
        self.Model.addLConstr(
            self.h_eff[p, d] >= self.App[p, d] - (1 - self.y[p, d]),
            name=f"h_eff_lower_{p}_{d}"
        )

    def _add_requirement_constraint(self, p, d, use_heff=False):
        """
        Add requirement fulfillment constraint.

        Args:
            p: Profile/patient index
            d: Day
            use_heff: If True, use h_eff (product of y and App), else use App directly
        """
        sessions = gu.quicksum(
            gu.quicksum(self.x[p, t, j, self.col_id] for j in range(self.Entry[p], d + 1))
            for t in self.T
        )

        if use_heff:
            learning_benefit = gu.quicksum(self.h_eff[p, j] for j in range(self.Entry[p], d + 1))
        else:
            learning_benefit = gu.quicksum(self.App[p, j] for j in range(self.Entry[p], d + 1))

        self.Model.addConstr(
            self.l[p, d] * self.Req[p] <= sessions + learning_benefit,
            name=f"requirement_{p}_{d}"
        )

    def genObj(self):
        """Generate objective function."""
        self.Model.setObjective(
            self.E[self.P] * self.LOS[self.P, self.col_id] -
            gu.quicksum(self.x[self.P, t, d, self.col_id] * self.duals_td[t, d]
                        for t in self.T for d in self.D) -
            self.duals_p[self.P],
            sense=gu.GRB.MINIMIZE
        )

    def getOptVals(self, var_name):
        """Get optimal values for a variable."""
        variable = getattr(self, var_name, None)
        if variable is None:
            raise AttributeError(f"Variable '{var_name}' not found in the class.")

        partial_dict = {idx: v.Xn for idx, v in variable.items()}
        values_dict = {}

        if var_name == "x":
            for t in self.T:
                day_range = self.D_p if self.reduction else self.D
                for d in day_range:
                    idx = (self.P, t, d, self.col_id)
                    values_dict[idx] = 0
            for idx, var in variable.items():
                values_dict[idx] = var.Xn

        elif var_name in ["y", "l", "S", "App", "w"]:
            day_range = self.D_p if self.reduction else self.D
            for d in day_range:
                idx = (self.P, d)
                values_dict[idx] = 0
            for idx, var in variable.items():
                values_dict[idx] = var.Xn

        elif var_name == "z":
            for t in self.T:
                idx = (self.P, t)
                values_dict[idx] = 0
            for idx, var in variable.items():
                values_dict[idx] = var.Xn

        elif var_name == "LOS":
            idx = (self.P, self.col_id)
            values_dict[idx] = 0
            for idx, var in variable.items():
                values_dict[idx] = var.Xn

        values_list = list(values_dict.values())
        return values_dict, values_list, partial_dict

    def getVarSol(self, var_name, current_itr):
        """Get variable solution with updated iteration index."""
        variable = getattr(self, var_name, None)
        if variable is None:
            raise AttributeError(f"Variable '{var_name}' not found in the class.")

        old_dict, _, _ = self.getOptVals(var_name)

        if var_name == "x":
            values_dict_unrounded = {(p, t, d, current_itr): val for (p, t, d, a), val in old_dict.items()}
            values_dict = {key: round(value) for key, value in values_dict_unrounded.items()}

        elif var_name in ["y", "l", "S"]:
            values_dict_unrounded = {(p, d, current_itr): val for (p, d), val in old_dict.items()}
            values_dict = {key: round(value) for key, value in values_dict_unrounded.items()}

        elif var_name == "App":
            values_dict = {(p, d, current_itr): val for (p, d), val in old_dict.items()}

        elif var_name == "LOS":
            old_dict = {(self.P, current_itr): 0}
            for idx, var in variable.items():
                old_dict[idx] = var.Xn
            values_dict = {(self.P, current_itr): round(var.Xn)}

        elif var_name == "z":
            values_dict_unrounded = {(p, t, current_itr): val for (p, t), val in old_dict.items()}
            values_dict = {key: round(value) for key, value in values_dict_unrounded.items()}

        return values_dict

    def solModel(self):
        """Solve the model."""
        self.Model.Params.MIPGap = 0.05
        self.Model.Params.MIPFocus = 1
        self.Model.Params.Heuristics = 0.5
        self.Model.Params.Cuts = 1
        self.Model.Params.Presolve = 2
        self.Model.Params.TimeLimit = 60
        self.Model.Params.IntFeasTol = 1e-5
        self.Model.Params.FeasibilityTol = 1e-6
        self.Model.Params.IntegralityFocus = 0
        self.Model.Params.OutputFlag = 0
        self.Model.optimize()

        if self.Model.status == gu.GRB.INFEASIBLE:
            boxed_print('\nThe following constraints and variables are in the IIS:')
            self.Model.computeIIS()
            for c in self.Model.getConstrs():
                if c.IISConstr:
                    boxed_print(f'\t{c.constrName}: {self.Model.getRow(c)} {c.Sense} {c.RHS}')
            for v in self.Model.getVars():
                if v.IISLB:
                    boxed_print(f'\t{v.varName} ≥ {v.LB}')
                if v.IISUB:
                    boxed_print(f'\t{v.varName} ≤ {v.UB}')

    def create_lambda_list(self, index):
        """Create lambda list for join patients."""
        if index in self.P_Join:
            ind = self.P_Join.index(index)
            lst = [0] * len(self.P_Join)
            lst[ind] = 1
            return lst
        return None