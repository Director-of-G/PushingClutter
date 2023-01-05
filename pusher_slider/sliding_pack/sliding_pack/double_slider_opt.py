# Author: Joao Moura (Modifed by Yongpeng Jiang)
# Contact: jpousad@ed.ac.uk (jyp19@mails.tsinghua.edu.cn)
# Date: 02/06/2021 (Modified on 12/20/2022)
# -------------------------------------------------------------------
# Description:
# 
# Class for the trajectory optimization (TO) for the single-pusher-
# double-slider problem using a Non-Linear Program (NLP) approach
# -------------------------------------------------------------------

# import libraries
import sys
import os
import time
import numpy as np
import casadi as cs
import sliding_pack

class buildDoubleSliderOptObj():

    def __init__(self, dyn_class, timeHorizon, configDict, X_nom_val=None,
                 controlRelPose=True, dt=0.1, maxIter=None):
        
        # init parameters
        self.dyn = dyn_class
        self.TH = timeHorizon
        self.solver_name = configDict['solverName']
        self.code_gen = configDict['codeGenFlag']
        self.controlRelPose = controlRelPose
        self.max_iter = maxIter

        # define weight matrix, which will be used in the cost function
        self.W_x = cs.diag(cs.SX(configDict['W_x']))[:self.dyn.Nx, :self.dyn.Nx]
        if self.controlRelPose is not True:
            self.W_x[:4, :4] = 0.
        self.W_u = cs.diag(cs.SX(configDict['W_u']))[:self.dyn.Nu, :self.dyn.Nu]
        self.K_goal = configDict['K_goal']
        if X_nom_val is None:
            self.X_nom_val = cs.DM.zeros(self.dyn.Nx, self.TH)
        else:
            self.X_nom_val = X_nom_val if (type(X_nom_val) == cs.DM) else cs.DM(X_nom_val)

        self.no_printing = configDict['noPrintingFlag']

        # opt var dimensionality
        self.Nxu = self.dyn.Nx + self.dyn.Nu
        self.Nxuzw = self.Nxu + self.dyn.Nz + self.dyn.Nw
        self.Nopt = self.Nxu + self.dyn.Nz + self.dyn.Nw + self.dyn.Ns

        # initialize variables for opt and args
        self.opt = sliding_pack.opt.OptVars()
        self.opt.x = []
        self.opt.g = []
        self.opt.f = []
        self.opt.p = []
        self.opt.discrete = []
        self.args = sliding_pack.opt.OptArgs()
        self.args.lbx = []
        self.args.ubx = []
        self.args.lbg = []
        self.args.ubg = []

        # define path variables
        self.X = cs.SX.sym('X', self.dyn.Nx, self.TH)
        self.U = cs.SX.sym('U', self.dyn.Nu, self.TH-1)
        self.Z = cs.SX.sym('Z', self.dyn.Nz, self.TH-1)
        self.W = cs.SX.sym('W', self.dyn.Nw, self.TH-1)
        self.X_nom = cs.SX.sym('X_nom', self.dyn.Nx, self.TH)
        self.X_bar = self.X - self.X_nom

        # initial state
        self.x0 = cs.SX.sym('x0', self.dyn.Nx)
        
        # slack variables
        self.Nphases = self.TH-1
        self.Sphases = cs.SX.sym('S', self.dyn.Ns, self.Nphases)
        self.S = self.Sphases  # built-in deepcopy
        
        # constraint functions
        #  -------------------------------------------------------------------
        # ---- Define Dynamic constraints ----
        __x_next = cs.SX.sym('__x_next', self.dyn.Nx)
        self.f_error = cs.Function(
                'f_error',
                [self.dyn.x_opt, self.dyn.u_opt, self.dyn.z_opt, __x_next, self.dyn.beta],
                [__x_next-self.dyn.x_opt-dt*self.dyn.f_opt(self.dyn.beta,self.dyn.x_opt,self.dyn.u_opt,self.dyn.z_opt)])
        
        # ---- Define Linear Constraints
        self.h_xuzw = cs.Function(
                'h_xuzw',
                [self.dyn.beta, self.dyn.x_opt, self.dyn.u_opt, self.dyn.z_opt, self.dyn.w_opt],
                [self.dyn.w_opt-self.dyn.q_opt(self.dyn.u_opt)-cs.mtimes(self.dyn.M_opt(self.dyn.beta, self.dyn.x_opt), self.dyn.z_opt)])
        
        # ---- Map dynamics constraint ----
        self.F_error = self.f_error.map(self.TH-1)
        #  -------------------------------------------------------------------
        # ---- Map linear constraints ----
        self.H_xuzw = self.h_xuzw.map(self.TH-1)
        #  -------------------------------------------------------------------
        # ---- Map complementary constraints ----
        self.G_zw = self.dyn.g_zw.map(self.TH-1)
        #  -------------------------------------------------------------------)

        #  -------------------------------------------------------------------
        # --- Define cost functions ----
        __x_bar = cs.SX.sym('x_bar', self.dyn.Nx)
        # self.cost_f = cs.Function(
        #         'cost_f',
        #         [self.dyn.x_opt, self.dyn.u_opt],
        #         [cs.dot(self.dyn.x_opt, cs.mtimes(self.W_x, self.dyn.x_opt)) + 
        #          cs.dot(self.dyn.u_opt, cs.mtimes(self.W_u, self.dyn.u_opt))])
        self.cost_f = cs.Function(
                'cost_f',
                [__x_bar, self.dyn.u_opt],
                [cs.dot(__x_bar, cs.mtimes(self.W_x, __x_bar)) + 
                 cs.dot(self.dyn.u_opt, cs.mtimes(self.W_u, self.dyn.u_opt))])
        # self.term_cost_f = cs.Function(
        #         'cost_f',
        #         [__x_bar, self.dyn.u_opt],
        #         [cs.dot(__x_bar, cs.mtimes(cs.mtimes(self.W_x, np.diag([1.0, 1.0, 0.0])), __x_bar)) + 
        #          cs.dot(self.dyn.u_opt, cs.mtimes(self.W_u, self.dyn.u_opt))])
        self.cost_F = self.cost_f.map(self.TH-1)
        # ------------------------------------------
        self.kx_F = self.dyn.kx_f.map(self.TH-1)
        index_x = np.arange(1, self.TH)
        self.Kx = self.kx_F(index_x).T
        
        self.ks_F = self.dyn.ks_f.map(self.TH-1)
        index_s = np.linspace(0, 1, self.TH-1)
        self.Ks = self.ks_F(index_s).T
        #  -------------------------------------------------------------------

        #  -------------------------------------------------------------------
        #  Building the Problem
        #  -------------------------------------------------------------------

        # ---- Set optimization variables ----
        for i in range(self.TH-1):
            # ---- Add States to optimization variables ---
            self.opt.x += self.X[:, i].elements()
            if i == 0: # expanding state constraint for 1st one (what trick is this?)
                self.args.lbx += [1.5*x for x in self.dyn.lbx]
                self.args.ubx += [1.5*x for x in self.dyn.ubx]
            else:
                self.args.lbx += self.dyn.lbx
                self.args.ubx += self.dyn.ubx
            self.opt.discrete += [False]*self.dyn.Nx
            # ---- Add Actions to optimization variables ---
            self.opt.x += self.U[:, i].elements()
            self.args.lbx += self.dyn.lbu
            self.args.ubx += self.dyn.ubu
            self.opt.discrete += [False]*self.dyn.Nu
            # ---- Add Complementarities to optimization variables ---
            self.opt.x += self.Z[:, i].elements()
            self.args.lbx += self.dyn.lbz
            self.args.ubx += self.dyn.ubz
            self.opt.discrete += [False]*self.dyn.Nz
            
            self.opt.x += self.W[:, i].elements()
            self.args.lbx += self.dyn.lbw
            self.args.ubx += self.dyn.ubw
            self.opt.discrete += [False]*self.dyn.Nw
        # ---- Add Terminal State to optimization variables ---
        self.opt.x += self.X[:, -1].elements()
        self.args.lbx += self.dyn.lbx
        self.args.ubx += self.dyn.ubx
        self.opt.discrete += [False]*self.dyn.Nx
        for i in range(self.Nphases):
            # ---- Add slack/additional opt variables ---
            self.opt.x += self.Sphases[:, i].elements()
            self.args.lbx += self.dyn.lbs
            self.args.ubx += self.dyn.ubs
            self.opt.discrete += [False]*self.dyn.Ns
        
        # ---- Set optimzation constraints ----
        self.opt.g = (self.X[:, 0]-self.x0).elements()  # Initial Conditions
        self.args.lbg = [0.0]*self.dyn.Nx
        self.args.ubg = [0.0]*self.dyn.Nx

        # ---- Dynamic constraints ----
        self.opt.g += self.F_error(
                self.X[:, :-1], self.U, self.Z,
                self.X[:, 1:],
                self.dyn.beta).elements()
        self.args.lbg += [0.] * self.dyn.Nx * (self.TH-1)
        self.args.ubg += [0.] * self.dyn.Nx * (self.TH-1)
        
        # ---- Linear constraints ----
        self.opt.g += self.H_xuzw(self.dyn.beta, self.X[:, :-1], self.U, self.Z, self.W).elements()
        self.args.lbg += [0.] * self.dyn.Nw * (self.TH-1)
        self.args.ubg += [0.] * self.dyn.Nw * (self.TH-1)
            
        # ---- Complementary constraints ----
        self.opt.g += self.G_zw(self.Z, self.W, self.S).elements()
        self.args.lbg += self.dyn.g_lb * (self.TH-1)
        self.args.ubg += self.dyn.g_ub * (self.TH-1)

        self.opt.f = cs.sum2(self.cost_F(self.X_bar[:, :-1], self.U))
        self.opt.f += self.K_goal*self.cost_f(self.X_bar[:, -1], self.U[:, -1])
        
        # penalize the state variables
        # for i in range(self.dyn.Nx):
        #     self.opt.f += cs.sum1(self.Kx*(self.X[i, 1:]-self.x0[i]).T**2)
        # self.opt.f += self.K_goal*self.Kx[0]*cs.sum2((self.X[:, -1]-self.x0).T**2)
        
        # in this way the problem with complementary constraints could be tackled easily
        for i in range(self.dyn.Ns):
            # self.opt.f += cs.sum1(self.Ks*(self.S[i].T**2))  # only punish the first timestep
            self.opt.f += cs.sum1(self.Ks*(self.S[i, :].T**2))  # punish all timesteps

        # ---- Set optimization parameters ----
        self.opt.p = []
        self.opt.p += self.dyn.beta.elements()
        self.opt.p += self.x0.elements()
        self.opt.p += self.X_nom.elements()

        # Set up QP Optimization Problem
        #  -------------------------------------------------------------------
        # ---- Set solver options ----
        opts_dict = {'print_time': 0}
        prog_name = 'MPC' + '_TH' + str(self.TH) + '_' + self.solver_name + '_codeGen_' + str(self.code_gen)
        if self.solver_name == 'ipopt':
            if self.no_printing: opts_dict['ipopt.print_level'] = 0
            opts_dict['ipopt.jac_d_constant'] = 'yes'
            opts_dict['ipopt.warm_start_init_point'] = 'yes'
            opts_dict['ipopt.hessian_constant'] = 'yes'
            opts_dict['ipopt.max_iter'] = self.max_iter
        if self.solver_name == 'knitro':
            opts_dict['knitro'] = {}
            # opts_dict['knitro.maxit'] = 80
            opts_dict['knitro.feastol'] = 1.e-3
            if self.no_printing: opts_dict['knitro']['mip_outlevel'] = 0
        if self.solver_name == 'snopt':
            opts_dict['snopt'] = {}
            if self.no_printing: opts_dict['snopt'] = {'Major print level': '0', 'Minor print level': '0'}
            opts_dict['snopt']['Hessian updates'] = 1
        if self.solver_name == 'qpoases':
            if self.no_printing: opts_dict['printLevel'] = 'none'
            opts_dict['sparse'] = True
        if self.solver_name == 'gurobi':
            if self.no_printing: opts_dict['gurobi.OutputFlag'] = 0
        # ---- Create solver ----
        # print('************************')
        # print(len(self.opt.x))
        # print(len(self.opt.g))
        # print(len(self.opt.p))
        # print('************************')
        prob = {'f': self.opt.f,
                'x': cs.vertcat(*self.opt.x),
                'g': cs.vertcat(*self.opt.g),
                'p': cs.vertcat(*self.opt.p)
                }
        # ---- add discrete flag ----
        opts_dict['discrete'] = self.opt.discrete  # add integer variables
        if (self.solver_name == 'ipopt') or (self.solver_name == 'snopt') or (self.solver_name == 'knitro'):
            self.solver = cs.nlpsol('solver', self.solver_name, prob, opts_dict)
            if self.code_gen:
                if not os.path.isfile('./' + prog_name + '.so'):
                    self.solver.generate_dependencies(prog_name + '.c')
                    os.system('gcc -fPIC -shared -O3 ' + prog_name + '.c -o ' + prog_name + '.so')
                self.solver = cs.nlpsol('solver', self.solver_name, prog_name + '.so', opts_dict)
        elif (self.solver_name == 'gurobi') or (self.solver_name == 'qpoases'):
            self.solver = cs.qpsol('solver', self.solver_name, prob, opts_dict)
        #  -------------------------------------------------------------------

    def solveProblem(self, idx, x0, beta, X_warmStart=None, U_warmStart=None):
        # ---- setting parameters ---- 
        p_ = []  # set to empty before reinitialize
        p_ += beta
        p_ += x0
        p_ += self.X_nom_val[:, idx:(idx+self.TH)].elements()  # Set x_nom to be the whole trajectory piece
        # p_ += cs.DM(self.X_nom_val[:, (idx+self.TH)-1].toarray().repeat(self.TH, 1)).elements()  # Set x_nom to be the goal point
        # ---- Set warm start ----
        self.args.x0 = []
        if (X_warmStart is not None) and (type(X_warmStart) is not cs.DM):
            X_warmStart = cs.DM(X_warmStart)
        if (U_warmStart is not None) and (type(U_warmStart) is not cs.DM):
            U_warmStart = cs.DM(U_warmStart)
        if X_warmStart is not None:
            for i in range(self.TH-1):
                # self.args.x0 += (np.array(x0)+(X_warmStart[:, idx+i]-X_warmStart[:, idx])).elements()
                self.args.x0 += X_warmStart[:, idx+i].elements()
                self.args.x0 += U_warmStart[:, idx+i].elements()
                self.args.x0 += [0.0]*self.dyn.Nz
                self.args.x0 += [0.0]*self.dyn.Nw
            # self.args.x0 += (np.array(x0)+(X_warmStart[:, (idx+self.TH)-1]-X_warmStart[:, idx])).elements()
            self.args.x0 += X_warmStart[:, (idx+self.TH)-1].elements()
            print(len(self.args.x0))
        else:
            for i in range(self.TH-1):
                # self.args.x0 += (x0+(self.X_nom_val[:, idx+i]-X_warmStart[:, idx])).elements()
                self.args.x0 += self.X_nom_val[:, idx+i].elements()
                self.args.x0 += [0.0]*self.dyn.Nu
                self.args.x0 += [0.0]*self.dyn.Nz
                self.args.x0 += [0.0]*self.dyn.Nw
            # self.args.x0 += (x0+(self.X_nom_val[:, (idx+self.TH)-1]-X_warmStart[:, idx])).elements()
            self.args.x0 += self.X_nom_val[:, (idx+self.TH)-1].elements()
        for i in range(self.Nphases):
            self.args.x0 += self.dyn.s0
        # import pdb; pdb.set_trace()
        # ---- Solve the optimization ----
        start_time = time.time()
        # print('-----------------------------')
        # print(len(self.args.x0))
        # print(len(self.args.lbx))
        # print(len(self.args.ubx))
        # print(len(self.args.lbg))
        # print(len(self.args.ubg))
        # print(len(p_))
        # print('-----------------------------')
        sol = self.solver(
                x0=self.args.x0,
                lbx=self.args.lbx, ubx=self.args.ubx,
                lbg=self.args.lbg, ubg=self.args.ubg,
                p=p_)
        # print(sol)
        # sys.exit()
        # ---- save computation time ---- 
        t_opt = time.time() - start_time
        # ---- decode solution ----
        resultFlag = self.solver.stats()['success']
        opt_sol = sol['x']
        f_opt = sol['f']
        # get x_opt, u_opt, other_opt
        opt_sol_xuzw = opt_sol[:(self.TH*self.Nxuzw-self.dyn.Nu-self.dyn.Nz-self.dyn.Nw)]
        opt_sol_s = opt_sol[(self.TH*self.Nxuzw-self.dyn.Nu-self.dyn.Nz-self.dyn.Nw):]
        x_opt = []
        for i in range(self.dyn.Nx):
            x_opt = cs.vertcat(x_opt, opt_sol_xuzw[i::self.Nxuzw].T)
        u_opt = []
        for i in range(self.dyn.Nx, self.Nxu):
            u_opt = cs.vertcat(u_opt, opt_sol_xuzw[i::self.Nxuzw].T)
        z_opt = []
        for i in range(self.Nxu, self.Nxu + self.dyn.Nz):
            z_opt = cs.vertcat(z_opt, opt_sol_xuzw[i::self.Nxuzw].T)
        w_opt = []
        for i in range(self.Nxu + self.dyn.Nz, self.Nxu + self.dyn.Nz + self.dyn.Nw):
            w_opt = cs.vertcat(w_opt, opt_sol_xuzw[i::self.Nxuzw].T)
        # get other_opt
        other_opt = []
        for i in range(self.dyn.Ns):
            other_opt = cs.vertcat(other_opt, opt_sol_s[i::self.dyn.Ns].T)
        # ---- warm start ----
        # for i in range(0, self.dyn.Nx):
        #     opt_sol[i::self.Nopt] = [0.0]*(self.TH)
        # clear the inputs, the inputs=0 in warm start solution
        for i in range(self.dyn.Nx, self.Nxuzw):
            opt_sol[:(self.TH*self.Nxuzw-self.dyn.Nu-self.dyn.Nz-self.dyn.Nw)][i::self.Nxuzw] = [0.]*(self.TH-1)
        # clear the slack variables
        opt_sol[(self.TH*self.Nxuzw-self.dyn.Nu-self.dyn.Nz-self.dyn.Nw):] = [0.]
        self.args.x0 = opt_sol.elements()

        return resultFlag, x_opt, u_opt, z_opt, w_opt, other_opt, f_opt, t_opt
