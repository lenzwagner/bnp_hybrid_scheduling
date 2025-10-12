import gurobipy as gu
import math
import time
from collections import defaultdict


class Problem_d:
    def __init__(self, df, Req, Entry, Max_t, app_data, pre_x, W_coeff, num_tangents, learn_method):
        self.P = df['P'].dropna().astype(int).unique().tolist()
        self.Entry = Entry
        self.D = df['D_Ext'].dropna().astype(int).unique().tolist()
        self.D_F = df['D'].dropna().astype(int).unique().tolist()
        self.P_Pre = [p for p, d in self.Entry.items() if d < min(self.D_F)]
        self.P_Focus = [p for p, d in self.Entry.items() if min(self.D_F) <= d <= max(self.D_F)]
        self.P_Post = [p for p, d in self.Entry.items() if d > max(self.D_F)]
        self.T = df['T'].dropna().astype(int).unique().tolist()
        self.Model = gu.Model("Compact")
        self.Req = Req
        self.P_Join = [p for p, d in self.Entry.items() if d >= min(self.D_F)]
        self.learn_method = learn_method
        self.pwl = True
        self.num_tangents = num_tangents
        self.W_coeff = W_coeff
        self.Max_t = Max_t
        self.pre_x = pre_x
        self.app_data = app_data
        self.M = len(self.D) + 1
        self.approx = False
        self.S_Bound = {p: max(10, math.ceil(min((self.Req[p] / self.W_coeff) + 2, max(self.D) - self.Entry[p]) * (
                    self.app_data["MS"][0] - self.app_data["MS_min"][0]) / self.app_data["MS"][0])) for p in
                        self.P_Join}
        self.aggregated = defaultdict(int)
        for (p, t, d), value in self.pre_x.items():
            self.aggregated[(t, d)] += value
        print(self.P, self.P_Join, self.P_Pre, self.P_Focus, self.P_Post, sep="\n")

    def buildModel(self):
        self.t0 = time.time()
        self.genVars()
        self.genCons()
        self.genLearnCons()
        self.genObj()
        self.Model.update()

    def genVars(self):
        self.l = self.Model.addVars(self.P_Join, self.D, vtype=gu.GRB.BINARY, name="l")
        self.LOS = self.Model.addVars(self.P_Join, vtype=gu.GRB.INTEGER, name="LOS")
        self.x = self.Model.addVars(self.P_Join, self.T, self.D, vtype=gu.GRB.BINARY, name="x")
        self.y = self.Model.addVars(self.P_Join, self.D, vtype=gu.GRB.BINARY, name="y")
        self.z = self.Model.addVars(self.P_Join, self.T, vtype=gu.GRB.BINARY, name="z")
        self.w = self.Model.addVars(self.P_Join, self.D, vtype=gu.GRB.BINARY, name="w")
        self.S = self.Model.addVars(self.P_Join, self.D, vtype=gu.GRB.INTEGER, name="S")
        self.Model.update()

    def solveStart(self):
        self.Model.Params.MIPGap = 0.9
        self.Model.update()
        self.Model.optimize()

    def genCons(self):
        for t in self.T:
            for d in self.D:
                self.Model.addLConstr(
                    gu.quicksum(self.x[p, t, d] for p in self.P_Join) + self.aggregated[t, d] <= self.Max_t[t, d],
                    name=f'demand({t, d})')
        for p in self.P_Join:
            self.Model.addLConstr(self.LOS[p] >= max(self.D) - self.Entry[p] + 1 - self.M * (
                gu.quicksum(self.l[p, d] for d in self.D)), name=f'LOS3({p})')
            self.Model.addLConstr(self.LOS[p] <= max(self.D) - self.Entry[p] + 1 + self.M * (
                gu.quicksum(self.l[p, d] for d in self.D)), name=f'LOS4({p})')
            if p in self.P_Focus:
                self.Model.addLConstr(gu.quicksum(self.l[p, d] for d in self.D) == 1, name=f'discharge_{p}')
            else:
                self.Model.addLConstr(gu.quicksum(self.l[p, d] for d in self.D) <= 1, name=f'no_discharge_{p}')
            self.Model.addLConstr(
                self.LOS[p] >= gu.quicksum(self.l[p, d] * d for d in self.D) - self.Entry[p] + 1 - self.M * (
                        1 - gu.quicksum(self.l[p, d] for d in self.D)), name=f'LOS1({p})')
            self.Model.addLConstr(
                self.LOS[p] <= gu.quicksum(self.l[p, d] * d for d in self.D) - self.Entry[p] + 1 + self.M * (
                        1 - gu.quicksum(self.l[p, d] for d in self.D)), name=f'LOS2({p})')
            self.Model.addLConstr(gu.quicksum(self.z[p, t] for t in self.T) == 1, name=f'single_therapist_{p}')
            self.Model.addLConstr(gu.quicksum(self.x[p, t, self.Entry[p]] for t in self.T) == 1,
                                  name=f'first_day_{p}')
            for d in range(self.Entry[p] + 1, max(self.D) + 1):
                self.Model.addLConstr(self.w[p, d] == 1 - gu.quicksum(self.l[p, k] for k in range(self.Entry[p], d)),
                                      name=f'last_day_{p}_{d}')
            for d in self.D:
                if d < self.Entry[p]:
                    self.Model.addLConstr(self.w[p, d] == 0, name=f'eligable1_{p}_{d}')
                elif d == self.Entry[p]:
                    self.Model.addLConstr(self.w[p, d] == 1, name=f'eligable2_{p}_{d}')
                self.Model.addLConstr(self.l[p, d] <= gu.quicksum(self.x[p, t, d] for t in self.T),
                                      name=f'therapist_last_day_{p}_{d}')
                self.Model.addLConstr(
                    self.w[p, d] == gu.quicksum(self.x[p, t, d] for t in self.T) + self.y[p, d],
                    name=f'assignment_{p}_{d}')
                if d < max(self.D) - self.app_data["MS"][0] + 1:
                    self.Model.addConstr(gu.quicksum(
                        self.x[p, t, j] for t in self.T for j in range(d, d + self.app_data["MS"][0])) >=
                                         self.app_data["MS_min"][0] - self.app_data["MS_min"][0] * (
                                                 self.app_data["MS"][0] - gu.quicksum(
                                             self.w[p, j] for j in range(d, d + self.app_data["MS"][0]))),
                                         name=f'MS_min_{p}_{d}')
                for t in self.T:
                    self.Model.addLConstr(self.x[p, t, d] <= self.z[p, t], name=f'single_therapist_2_{p}')
            self.Model.update()

    def genLearnCons(self):
        if self.app_data["learn_type"][0] in ['exp', 'sigmoid'] and self.learn_method == 'approx':
            self.App = self.Model.addVars(self.P_Join, self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="App")

            k_learn = self.app_data["k_learn"][0]
            theta_base = self.app_data["theta_base"][0]

            if self.app_data["learn_type"][0] in ['exp']:
                patient_tangents = {}
                for p in self.P_Join:
                    Y_bound_p = self.S_Bound[p]
                    patient_tangents[p] = self.compute_tangent_approximation(k_learn, theta_base, Y_bound_p)
            else:
                infl_point = self.app_data["infl_point"][0]
                patient_tangents = {}
                for p in self.P_Join:
                    Y_bound_p = self.S_Bound[p]
                    patient_tangents[p] = self.compute_tangent_approximation_sigmoid(k_learn, infl_point, theta_base,
                                                                                     Y_bound_p)

            for p in self.P_Join:
                for d in self.D:
                    self.Model.addLConstr(
                        self.S[p, d] == gu.quicksum(self.y[p, t] for t in range(self.Entry[p], d + 1)),
                        name=f"define_Y_{p}_{d}")
                    for s, (m_s, beta_s, Y_s) in enumerate(patient_tangents[p]):
                        self.Model.addLConstr(self.App[p, d] <= m_s * self.S[p, d] + beta_s,
                                              name=f"tangent_{p}_{d}_{s}")
                    self.Model.addLConstr(self.App[p, d] <= 1, name=f"App_ub_{p}_{d}")
                    self.Model.addConstr(self.l[p, d] * self.Req[p] <= gu.quicksum(
                        gu.quicksum(self.x[p, t, j] for j in range(self.Entry[p], d + 1)) for t in self.T) +
                                         gu.quicksum(self.App[p, j] for j in range(self.Entry[p], d + 1)),
                                         name=f"equivalent_{p}_{d}")

        elif self.app_data["learn_type"][0] in ['exp', 'sigmoid'] and self.learn_method == 'pwl':
            self.h_eff = self.Model.addVars(self.P_Join, self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="h_eff")
            self.App = self.Model.addVars(self.P_Join, self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="h_eff")
            self.S = self.Model.addVars(self.P_Join, self.D, vtype=gu.GRB.CONTINUOUS, lb=0, name="S")

            k, x0 = self.app_data["k_learn"][0], self.app_data["infl_point"][0]

            for p in self.P_Join:
                for d in self.D:
                    R = list(range(1, 1 + self.S_Bound[p]))
                    R_sorted = sorted(list(R))
                    x_values = R_sorted

                    if self.app_data["learn_type"][0] == 'exp':
                        y_values = [self.app_data["theta_base"][0] +
                                    (1 - self.app_data["theta_base"][0]) * (1 - math.exp(-k * r))
                                    for r in R_sorted]
                    elif self.app_data["learn_type"][0] == 'sigmoid':
                        y_values = [self.app_data["theta_base"][0] +
                                    (1 - self.app_data["theta_base"][0]) / (1 + math.exp(-k * (r - x0)))
                                    for r in R_sorted]

                    self.Model.addConstr(self.S[p, d] == gu.quicksum(self.y[p, t] for t in range(self.Entry[p], d + 1)),
                                         name=f"define_S_{p}_{d}")
                    self.Model.addGenConstrPWL(self.S[p, d], self.App[p, d], x_values, y_values,
                                               name=f"pwl_App_{p}_{d}")
                    self.Model.addConstr(self.h_eff[p, d] <= self.y[p, d], name=f"h_eff_y_{p}_{d}")
                    self.Model.addConstr(self.h_eff[p, d] <= self.App[p, d], name=f"h_eff_theta_{p}_{d}")
                    self.Model.addConstr(self.h_eff[p, d] >= self.App[p, d] - (1 - self.y[p, d]),
                                         name=f"h_eff_linear_{p}_{d}")
                    self.Model.addConstr(self.l[p, d] * self.Req[p] <= gu.quicksum(
                        gu.quicksum(self.x[p, t, j] for j in range(self.Entry[p], d + 1)) for t in
                        self.T) + gu.quicksum(self.h_eff[p, j] for j in range(self.Entry[p], d + 1)),
                                         name=f"equivalent_{p}_{d}")

        elif self.app_data["learn_type"][0] in ['exp', 'sigmoid']:
            self.h_eff = self.Model.addVars(self.P_Join, self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="h_eff")
            self.App = self.Model.addVars(self.P_Join, self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="App")
            k, infl_point = self.app_data["k_learn"][0], self.app_data["infl_point"][0]

            for p in self.P_Join:
                self.z_pdr = self.Model.addVars(self.P_Join, self.D, list(range(0, self.S_Bound[p] + 1)),
                                                vtype=gu.GRB.BINARY, name="z_pdr")
                exp_values = {
                    r: self.app_data["theta_base"][0] + (1 - self.app_data["theta_base"][0]) * (1 - math.exp(-k * r))
                    for r in list(range(0, self.S_Bound[p] + 1))}
                log_values = {r: self.app_data["theta_base"][0] + (1 - self.app_data["theta_base"][0]) / (
                            1 + math.exp(-k * (r - infl_point))) for r in list(range(0, self.S_Bound[p] + 1))}
                for d in self.D:
                    self.Model.addLConstr(
                        gu.quicksum(self.z_pdr[p, d, r] for r in list(range(0, self.S_Bound[p] + 1))) == 1,
                        name=f"sum_z_one_{p}_{d}")
                    self.Model.addLConstr(gu.quicksum(self.y[p, t] for t in range(self.Entry[p], d + 1)) == gu.quicksum(
                        r * self.z_pdr[p, d, r] for r in list(range(0, self.S_Bound[p] + 1))), name=f"define_S_{p}_{d}")
                    if self.app_data["learn_type"][0] == 'exp':
                        self.Model.addLConstr(self.App[p, d] == gu.quicksum(
                            exp_values[r] * self.z_pdr[p, d, r] for r in list(range(0, self.S_Bound[p] + 1))),
                                              name=f"define_theta_exp_{p}_{d}")
                    elif self.app_data["learn_type"][0] == 'sigmoid':
                        self.Model.addLConstr(self.App[p, d] == gu.quicksum(
                            log_values[r] * self.z_pdr[p, d, r] for r in list(range(0, self.S_Bound[p] + 1))),
                                              name=f"define_theta_sig_{p}_{d}")
                    self.Model.addLConstr(self.h_eff[p, d] <= self.y[p, d], name=f"h_eff_y_{p}_{d}")
                    self.Model.addLConstr(self.h_eff[p, d] <= self.App[p, d], name=f"h_eff_theta_{p}_{d}")
                    self.Model.addLConstr(self.h_eff[p, d] >= self.App[p, d] - (1 - self.y[p, d]),
                                          name=f"h_eff_linear_{p}_{d}")
                    self.Model.addConstr(self.l[p, d] * self.Req[p] <= gu.quicksum(
                        gu.quicksum(self.x[p, t, j] for j in range(self.Entry[p], d + 1)) for t in
                        self.T) + gu.quicksum(self.h_eff[p, j] for j in range(self.Entry[p], d + 1)),
                                         name=f"equivalent_{p}_{d}")

        elif self.app_data["learn_type"][0] == 'lin':
            self.App = self.Model.addVars(self.P_Join, self.D, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="App")
            for p in self.P_Join:
                for d in self.D:
                    self.Model.addLConstr(
                        self.S[p, d] == gu.quicksum(self.y[p, t] for t in range(self.Entry[p], d + 1)))
                    self.Model.addLConstr(
                        self.App[p, d] <= self.app_data["theta_base"][0] + self.app_data["lin_increase"][0] * self.S[
                            p, d])
                    self.Model.addLConstr(self.App[p, d] <= 1)
                    self.Model.addConstr(self.l[p, d] * self.Req[p] <= gu.quicksum(
                        gu.quicksum(self.x[p, t, j] for j in range(self.Entry[p], d + 1)) for t in
                        self.T) + gu.quicksum(self.App[p, j] for j in range(self.Entry[p], d + 1)),
                                         name=f"equivalent_{p}_{d}")

        else:
            for p in self.P_Join:
                for d in self.D:
                    self.Model.addConstr(self.l[p, d] * self.Req[p] <= gu.quicksum(
                        gu.quicksum(self.x[p, t, j] for j in range(self.D[0], d + 1)) for t in
                        self.T) + gu.quicksum(
                        self.y[p, j] * self.app_data["learn_type"][0] for j in range(self.Entry[p], d + 1)))
        self.Model.update()

    def genObj(self):
        self.Model.setObjective(gu.quicksum(self.LOS[p] for p in self.P_Focus), gu.GRB.MINIMIZE)

    def ModelParams(self):
        self.Model.setParam('ConcurrentMIP', 2)

    def solveModel(self):
        self.t1 = time.time()
        try:
            self.Model.Params.MIPGap = 0
            print("Start optimization")
            self.genObj()
            self.Model.update()
            self.Model.optimize()
            if self.Model.status == gu.GRB.OPTIMAL:
                print(f"Optimal Objective Value: {self.Model.objVal}")
            else:
                print("No optimal solution found.")
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

    def setStart(self, start_dict):
        for key, value in start_dict.items():
            self.x[key].Start = value
        self.Model.Params.MIPFocus = 3
        self.Model.update()

    def compute_tangent_approximation(self, k_learn, theta_base, Y_bound):
        """
        Calculates tangents for the outer approximation of the exponential learning curve.

        f(Y) = theta_base + (1 - theta_base) * (1 - exp(-k_learn * Y))
        f'(Y) = (1 - theta_base) * k_learn * exp(-k_learn * Y)

        Tangent at Y^(s): θ ≤ m_s * Y + β_s
        where: m_s = f'(Y^(s))
               β_s = f(Y^(s)) - m_s * Y^(s)
        """
        if Y_bound == 0 or self.num_tangents == 1:
            support_points = [0]
        else:
            support_points = [i * Y_bound / (self.num_tangents - 1) for i in range(self.num_tangents)]

        tangents = []
        for Y_s in support_points:
            f_Y_s = theta_base + (1 - theta_base) * (1 - math.exp(-k_learn * Y_s))
            m_s = (1 - theta_base) * k_learn * math.exp(-k_learn * Y_s)
            beta_s = f_Y_s - m_s * Y_s
            tangents.append((m_s, beta_s, Y_s))

        return tangents

    def compute_tangent_approximation_sigmoid(self, k_learn, infl_point, theta_base, Y_bound):
        """
        Calculates tangents for the outer approximation of the sigmoid learning curve.

        f(Y) = theta_base + (1 - theta_base) / (1 + exp(-k_learn * (Y - infl_point)))
        f'(Y) = (1 - theta_base) * k_learn * exp(-k_learn * (Y - infl_point)) / (1 + exp(-k_learn * (Y - infl_point)))^2
        """
        if Y_bound == 0 or self.num_tangents == 1:
            support_points = [0]
        else:
            support_points = [i * Y_bound / (self.num_tangents - 1) for i in range(self.num_tangents)]

        tangents = []
        for Y_s in support_points:
            exp_term = math.exp(-k_learn * (Y_s - infl_point))
            f_Y_s = theta_base + (1 - theta_base) / (1 + exp_term)
            m_s = (1 - theta_base) * k_learn * exp_term / ((1 + exp_term) ** 2)
            beta_s = f_Y_s - m_s * Y_s
            tangents.append((m_s, beta_s, Y_s))

        return tangents

    def printLOS(self):
        for p in self.P_Focus:
            print(f"Patient {p}: LOS = {self.LOS[p].X:.2f}")
        return None