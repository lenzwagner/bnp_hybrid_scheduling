import gurobipy as gu
from Utils.Generell.utils import *

class MasterProblem_d:
    def __init__(self, df, T_Max, Nr_agg, Req, pre_x, E_dict):
        self.P_Full = df['P_Full'].dropna().astype(int).unique().tolist()
        self.P_Pre = df['P_Pre'].dropna().astype(int).unique().tolist()
        self.P_Join = df['P_Join'].dropna().astype(int).unique().tolist()
        self.P_Focus = df['P_F'].dropna().astype(int).unique().tolist()
        self.P_Post = df['P_Post'].dropna().astype(int).unique().tolist()
        self.D = df['D_Ext'].dropna().astype(int).unique().tolist()
        self.T = df['T'].dropna().astype(int).unique().tolist()
        self.G = df['G'].dropna().astype(int).unique().tolist()
        self.A = [1]
        self._solve_counter = 0
        self.beta = 0.5
        self.Model = gu.Model("MasterProblem")
        self.cons_p_max = {}
        self.cons_los = {}
        self.E = E_dict
        self.cons_lmbda = {}
        self.all_schedules = {}
        self.all_los = {}
        self.Nr = Nr_agg
        self.Req = Req
        self.output_len = 100
        self.T_max = T_Max
        self.pre_x = pre_x
        self.zero_count = {(p, a): 0 for p in self.P_Join for a in self.A}
        self.drop_threshold = 5
        self.column_pool = {}
        self.branching_bounds = {}
        self.aggregated = defaultdict(int)
        for (p, t, d), value in self.pre_x.items():
            self.aggregated[(t, d)] += value

    def buildModel(self):
        self.genVars()
        self.genCons()
        self.genObj()
        self.Model.update()

    def genVars(self):
        self.lmbda = self.Model.addVars(self.P_Join, self.A, vtype=gu.GRB.INTEGER, name='lmbda')


    def genCons(self):
        for p in self.P_Join:
            self.cons_lmbda[p] = self.Model.addConstr(gu.quicksum(self.lmbda[p, a] for a in self.A) == self.Nr[p], name=f"lambda({p})")

        for t in self.T:
            for d in self.D:
                pre_load = sum(self.pre_x.get((p, t, d), 0) for p in self.P_Pre)
                self.cons_p_max[t, d] = self.Model.addConstr(gu.quicksum(self.lmbda[p, a] * self.all_schedules.get((p, t, d, a), 0) for p in self.P_Join for a in self.A) + pre_load <= self.T_max[t, d], name=f"p_max({t},{d})")


    def genObj(self):
        self.Model.setObjective(gu.quicksum(self.lmbda[p, a] for p in self.P_Focus for a in self.A), sense=gu.GRB.MINIMIZE)

    def getDuals(self):
        return {(t, d): self.cons_p_max[t, d].Pi for t in self.T for d in self.D}, {p: self.cons_lmbda[p].Pi for p in self.P_Join}

    def addSchedule(self, schedule):
        self.all_schedules.update(schedule)

    def addLOS(self, schedule):
        self.all_los.update(schedule)

    def startSol(self, schedule_x = None, schedule_los = None):
        for p in self.P_Join:
            self.lmbda[p, 1].Obj = schedule_los.get(p, 0)
            for t, d in self.cons_p_max:
                value = schedule_x.get((p, t, d, 1), 0)
                self.Model.chgCoeff(self.cons_p_max[t, d], self.lmbda[p, 1], value)
        self.Model.update()

    def addLambdaVar(self, p, a, col, coef):
        new_col = gu.Column(col, self.Model.getConstrs())
        self.lmbda[p, a] = self.Model.addVar(obj=coef[0], vtype=gu.GRB.INTEGER, column=new_col, name=f"lmbda[{p},{a}]")
        self.A.append(a)
        self.zero_count[(p, a)] = 0
        self.Model.update()

    def finSol(self):
        all_integer, obj, most_frac_info = self.check_fractionality()
        self.Model.Params.OutputFlag = 1
        for var in self.lmbda.values():
            var.VType = gu.GRB.INTEGER
        self.Model.optimize()
        if self.Model.status == gu.GRB.INFEASIBLE:
            boxed_print('\nThe following constraints and variables are in the IIS:')
            self.Model.computeIIS()
            for c in self.Model.getConstrs():
                if c.IISConstr: boxed_print(f'\t{c.constrName}: {self.Model.getRow(c)} {c.Sense} {c.RHS}')
            for v in self.Model.getVars():
                if v.IISLB: boxed_print(f'\t{v.varName} ≥ {v.LB}')
                if v.IISUB: boxed_print(f'\t{v.varName} ≤ {v.UB}')
        status_msg = "Optimal solution found" if self.Model.status == gu.GRB.OPTIMAL else "No optimal solution found"
        boxed_print(f"*{'***** ' + status_msg + ' *****':^{self.output_len}}*")

        return all_integer, obj, most_frac_info

    def solRelModel(self):
        self.Model.Params.OutputFlag = 1
        self.Model.Params.Method = 2
        self._solve_counter += 1

        for var in self.Model.getVars():
            var.VType = gu.GRB.CONTINUOUS

            # ✅ Restore branching bounds if they exist
            var_name = var.VarName
            if var_name in self.branching_bounds:
                var.LB = self.branching_bounds[var_name]['lb']
                var.UB = self.branching_bounds[var_name]['ub']
                if self._solve_counter == 2:  # Only print once per node
                    print(f"    [Branching Bound] Restored {var_name}: "
                          f"LB={var.LB}, UB={var.UB}")
            else:
                # No branching bound - set default
                var.LB = 0.0
                # UB remains as is (usually infinity)

        if self._solve_counter > 1:
            self.Model.Params.LPWarmStart = 2

        self.Model.optimize()
        if self.Model.status != gu.GRB.OPTIMAL:
            boxed_print('\nThe following constraints and variables are in the IIS:')
            self.Model.computeIIS()
            for c in self.Model.getConstrs():
                if c.IISConstr: boxed_print(f'\t{c.constrName}: {self.Model.getRow(c)} {c.Sense} {c.RHS}')
            for v in self.Model.getVars():
                if v.IISLB: boxed_print(f'\t{v.varName} ≥ {v.LB}')
                if v.IISUB: boxed_print(f'\t{v.varName} ≤ {v.UB}')

    def check_fractionality(self):
        """
        Check if all lambda variables are integer and find the most fractional solution.
        Tie-break: (1) smallest n, (2) smallest a

        Returns:
            tuple: (all_integer: bool, obj_val: float, most_frac_info: dict or None)
        """
        import math

        self.solRelModel()
        self.Model.write('Frac.sol')
        obj = self.Model.ObjVal

        all_integer = True
        max_fractionality = 0.0
        most_frac_info = None

        # Check all lambda variables
        print(f'Lambda items: {len(self.lmbda.items())} variables')
        for (n, a), var in self.lmbda.items():
            x_val = var.X

            # Berechne Distanzen zu floor und ceil
            floor_val = math.floor(x_val)
            ceil_val = math.ceil(x_val)
            dist_to_floor = x_val - floor_val
            dist_to_ceil = ceil_val - x_val

            # Fraktionalität ist die minimale Distanz zum nächsten Integer
            frac_part = min(dist_to_floor, dist_to_ceil)

            if frac_part > 1e-8:
                all_integer = False

                # Bestimme ob diese Variable die neue "most fractional" ist
                is_new_most_frac = False

                if frac_part > max_fractionality + 1e-10:
                    # Höhere Fraktionalität gefunden
                    is_new_most_frac = True
                elif abs(frac_part - max_fractionality) < 1e-10:
                    # Gleiche Fraktionalität - Tie-Break anwenden
                    if most_frac_info is not None:
                        if n < most_frac_info['n']:
                            # Kleinerer n-Wert
                            is_new_most_frac = True
                        elif n == most_frac_info['n'] and a < most_frac_info['a']:
                            # Gleicher n-Wert, aber kleinerer a-Wert
                            is_new_most_frac = True
                    else:
                        # Erste fraktionale Variable
                        is_new_most_frac = True

                if is_new_most_frac:
                    max_fractionality = frac_part
                    most_frac_info = {
                        'n': n,
                        'a': a,
                        'value': x_val,
                        'fractionality': frac_part,
                        'floor': floor_val,
                        'ceil': ceil_val,
                        'dist_to_floor': dist_to_floor,
                        'dist_to_ceil': dist_to_ceil,
                        'var_name': var.VarName
                    }

        # Print results
        if all_integer:
            boxed_print("All lambda variables are integer.")
        else:
            boxed_print(f"Fractional solution detected!")
            if most_frac_info:
                print(f"\nMost fractional variable (tie-break: smallest n, then smallest a):")
                print(f"  Variable: lmbda[n={most_frac_info['n']}, a={most_frac_info['a']}]")
                print(f"  Value (X): {most_frac_info['value']:.6f}")
                print(f"  Fractionality: {most_frac_info['fractionality']:.6f}")
                print(f"  Floor: {most_frac_info['floor']}, Distance to floor: {most_frac_info['dist_to_floor']:.6f}")
                print(f"  Ceil:  {most_frac_info['ceil']}, Distance to ceil:  {most_frac_info['dist_to_ceil']:.6f}")

        return all_integer, obj, most_frac_info

    def finalDicts(self, sols_dict, app_data, with_post = None):
        active_keys = []
        models = [self.Model]

        for model in models:
            for v in model.getVars():
                if 'lmbda' in v.VarName and v.X > 0:
                    parts = v.VarName.split('[')[1].split(']')[0].split(',')
                    p = int(parts[0])
                    s = int(parts[1])
                    if with_post == None:
                        if p in self.P_Focus:
                            solution_key = (p, s)
                            active_keys.append(solution_key)
                            if v.Obj > 1e-2:
                                print(f'{v.VarName} = {v.X}, Obj-Coefficient: {round(v.Obj, 2)}')
                    else:
                        if p in self.P_Join:
                            solution_key = (p, s)
                            active_keys.append(solution_key)

        if isinstance(app_data, (int, float)):
            active_solutions = {'x': {}, 'LOS': {}, 'y': {}, 'z': {}, 'S': {}, 'l': {}}
        else:
            active_solutions = {'x': {}, 'LOS': {}, 'y': {}, 'z': {}, 'App': {}, 'S': {}, 'l': {}}
        for key in active_keys:
            if key in sols_dict['x']:
                active_solutions['x'].update(sols_dict['x'][key])
                active_solutions['LOS'].update(sols_dict['LOS'][key])
                active_solutions['y'].update(sols_dict['y'][key])
                active_solutions['z'].update(sols_dict['z'][key])
                active_solutions['S'].update(sols_dict['S'][key])
                active_solutions['l'].update(sols_dict['l'][key])
                if not isinstance(app_data, (int, float)):
                    active_solutions['App'].update(sols_dict['App'][key])
        return active_solutions


    def calculate_impact_score(self, reduced_cost, x_values, duals_td):
        """
        Calculates the impact score for a column

        Args:
            reduced_cost: reduced cost value of the column
            x_values: Assignment variables (x_{ptd}) of the column
            duals_td: Dual variables of the master problem for capacity constraints

        Returns:
            Impact Score = |reduced_cost| / (Capacity utilisation weighted with dual values)
        """
        capacity_usage = 0.0
        for (t, d), value in x_values.items():
            if value > 0.5:
                capacity_usage += duals_td.get((t, d), 0) * value

        if capacity_usage < 1e-10:
            capacity_usage = 1e-10

        return abs(reduced_cost) / capacity_usage

    def printLambda(self):
        models = [self.Model]
        active_keys = []

        for model in models:
            for v in model.getVars():
                if 'lmbda' in v.VarName and v.X > 0:
                    parts = v.VarName.split('[')[1].split(']')[0].split(',')
                    p = int(parts[0])
                    s = int(parts[1])
                    if p in self.P_Focus:
                        solution_key = (p, s)
                        active_keys.append(solution_key)
                        if v.Obj > 1e-2:
                            print(f'{v.VarName} = {v.X}, Obj-Coefficient: {round(v.Obj, 2)}')
                    else:
                        solution_key = (p, s)
                        active_keys.append(solution_key)
        return None

    def set_branching_bound(self, var, bound_type, value):
        """
        Set a branching bound that will be preserved during LP relaxation.

        Args:
            var: Gurobi variable
            bound_type: 'lb' or 'ub'
            value: Bound value
        """
        var_name = var.VarName

        if var_name not in self.branching_bounds:
            self.branching_bounds[var_name] = {'lb': 0.0, 'ub': float('inf')}

        self.branching_bounds[var_name][bound_type] = value

        # Set the bound on the variable
        if bound_type == 'lb':
            var.LB = value
        else:
            var.UB = value

        print(f"    [Branching Bound] Set {var_name}.{bound_type.upper()} = {value}")
