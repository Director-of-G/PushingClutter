# Dynamic model for two sliders with face-to-edge contact

import time

from matplotlib import pyplot as plt
import numpy as np
import casadi as cs

import sliding_pack


class Double_Slider_lim_surf_test_dyn():
    def __init__(self, configDict) -> None:
        # init parameters
        # -----------------------
        try:
            self.dt = configDict['timeInterval']
        except:
            self.dt = 0.04
        self.miu = configDict['pusherFricCoef']
        self.f_lim = 0.02
        self.v_lim = 0.1
        self.x_length = configDict['xLenght']
        self.y_length = configDict['yLenght']
        self.Ks_max = 50.
        self.Ks_min = 0.1
        
        # geometry and kinematics
        # -----------------------
        __xl = cs.SX.sym('__xl')  # slider x lenght
        __yl = cs.SX.sym('__yl')  # slider y lenght
        __r_pusher = cs.SX.sym('__r_pusher')  # radious of the cilindrical pusher
        __beta = cs.veccat(__xl, __yl, __r_pusher)
        __Area = __xl*__yl
        __int_Area = sliding_pack.integral.rect_cs(__xl, __yl)
        __c = __int_Area/__Area # ellipsoid approximation ratio
        self.c = cs.Function('c', [__beta], [__c], ['b'], ['c'])
        __A = cs.SX.sym('__A', cs.Sparsity.diag(3))
        __A[0,0] = __A[1,1] = 1.; __A[2,2] = 1./(__c**2)
        self.A = cs.Function('A', [__beta], [__A], ['b'], ['A'])
        
        __thetaA = cs.SX.sym('__thetaA')
        __thetaB = cs.SX.sym('__thetaB')
        __RA, __RB = cs.SX(3, 3), cs.SX(3, 3)
        __cthetaA = cs.cos(__thetaA); __sthetaA = cs.sin(__thetaA)
        __RA[0,0] = __cthetaA; __RA[0,1] = -__sthetaA; __RA[1,0] = __sthetaA; __RA[1,1] = __cthetaA; __RA[2,2] = 1.0
        __cthetaB = cs.cos(__thetaB); __sthetaB = cs.sin(__thetaB)
        __RB[0,0] = __cthetaB; __RB[0,1] = -__sthetaB; __RB[1,0] = __sthetaB; __RB[1,1] = __cthetaB; __RB[2,2] = 1.0
        self.RA = cs.Function('RA', [__thetaA], [__RA], ['t'], ['ra'])
        self.RB = cs.Function('RB', [__thetaB], [__RB], ['t'], ['rb'])
        
        __dtheta = cs.SX.sym('dtheta')
        __TB2A, __TA2B = cs.SX(2, 2), cs.SX(2, 2)
        __cdtheta = cs.cos(__dtheta); __sdtheta = cs.sin(__dtheta)
        __TB2A[0,0] = __cdtheta; __TB2A[0,1] = -__sdtheta; __TB2A[1,0] = __sdtheta; __TB2A[1,1] = __cdtheta
        __TA2B[0,0] = __cdtheta; __TA2B[0,1] = __sdtheta; __TA2B[1,0] = -__sdtheta; __TA2B[1,1] = __cdtheta
        
        # variables
        # -----------------------
        # contact force
        __fnA = cs.SX.sym('fnA')
        __ftA = cs.SX.sym('ftA')
        __fnB = cs.SX.sym('fnB')
        __ftB = cs.SX.sym('ftB')
        __f_ctact = cs.vertcat(__fnA, __ftA, __fnB, __ftB)
        
        # contact location
        __yctA = cs.SX.sym('yctA')
        __yctB = cs.SX.sym('yctB')
        __y_ctact = cs.vertcat(__yctA, __yctB)
        __vyctA = cs.SX.sym('vyctA')
        
        __JA0, __JA1, __JB0 = cs.SX(2, 3), cs.SX(2, 3), cs.SX(2, 3)
        __JA0[0, 0] = 1; __JA0[1, 1] = 1; __JA0[0, 2] = -__yctA; __JA0[1, 2] = -0.5*self.x_length
        __JA1[0, 0] = 1; __JA1[1, 1] = 1; __JA1[0, 2] = 0.5*self.y_length; __JA1[1, 2] = 0.5*self.x_length
        __JB0[0, 0] = 1; __JB0[1, 1] = 1; __JB0[0, 2] = -__yctB; __JB0[1, 2] = -0.5*self.x_length
        
        __VB = cs.mtimes(__A, cs.mtimes(__JB0.T, __f_ctact[2:4]))
        __VA = cs.mtimes(__A, cs.mtimes(__JA0.T, __f_ctact[0:2]) + cs.mtimes(__JA1.T, -cs.mtimes(__TB2A, __f_ctact[2:4])))

        __vctact = cs.mtimes(__TA2B, cs.mtimes(__JA1, __VA)) - cs.mtimes(__JB0, __VB)
        self.vctact = cs.Function('vctact', [__dtheta, __y_ctact, __f_ctact, __beta], [__vctact], ['dt', 'y', 'f', 'b'], ['v'])
        
        # dynamic equations
        __x_init = cs.SX.sym('x_init', 8)
        __x_new = cs.SX.sym('x_new', 8)
        __xA_new = __x_init[0:3] + cs.mtimes(__RA, __VA) * self.dt
        __yctB_new = __x_init[7] + __vctact[1] * self.dt
        __x_new[0:3] = __xA_new
        __x_new[3] = __x_init[3] + __vyctA * self.dt
        __x_new[6] = __x_init[6] + __VB[2] * self.dt
        # __RA_new, __RB_new = cs.SX(2, 2), cs.SX(2, 2)
        # __RA_new[0, 0] = cs.cos(__x_new[2]); __RA_new[0, 1] = -cs.sin(__x_new[2]); __RA_new[1, 0] = cs.sin(__x_new[2]); __RA_new[1, 1] = cs.cos(__x_new[2])
        # __RB_new[0, 0] = cs.cos(__x_new[6]); __RB_new[0, 1] = -cs.sin(__x_new[6]); __RB_new[1, 0] = cs.sin(__x_new[6]); __RB_new[1, 1] = cs.cos(__x_new[6])
        # __x_new[4:6] = __xA_new[0:2] + cs.mtimes(__RA_new, cs.vertcat(0.5*self.x_length, -0.5*self.y_length)) + cs.mtimes(__RB_new, -cs.vertcat(-0.5*self.x_length, __yctB_new))
        __x_new[4:6] = __xA_new[0:2] + cs.mtimes(__RA[0:2, 0:2], cs.vertcat(0.5*self.x_length, -0.5*self.y_length)) + cs.mtimes(__RB[0:2, 0:2], -cs.vertcat(-0.5*self.x_length, __yctB_new))
        __x_new[7] = __yctB_new
        # __x_new[4:8] = __x_init[4:8]
        
        __f = cs.Function('f', [__x_init, __thetaA, __thetaB, __dtheta, __y_ctact, __vyctA, __f_ctact, __beta],
                          [__x_new],
                          ['x_i', 'tA', 'tB', 'dt', 'y', 'vy', 'f', 'b'],
                          ['x_n'])
        
        # optimization
        # -----------------------
        self.Nx = 8
        self.x_opt = cs.SX.sym('x', 8)
        self.lbx = [-cs.inf]*3 + [-0.5*self.y_length] + [-cs.inf]*3 + [-0.5*self.y_length]
        self.ubx = [cs.inf]*3 + [0.5*self.y_length] + [cs.inf]*3 + [0.5*self.y_length]
        
        self.Nu = 6
        self.u_opt = cs.SX.sym('u', 6)
        self.lbu = [0., -self.f_lim, 0., 0., 0., 0.]
        self.ubu = [self.f_lim, self.f_lim, self.f_lim, self.f_lim, self.v_lim, self.v_lim]
        # self.lbu = [0., -self.f_lim, 0., 0., 0., 0.]
        # self.ubu = [self.f_lim, self.f_lim, 0., 0., self.v_lim, self.v_lim]
        
        self.Ns = 2
        self.s_opt = cs.SX.sym('s', 2)
        self.s0 = [1.]*2
        self.lbs = [-cs.inf]*2
        self.ubs = [cs.inf]*2
        
        self.beta = cs.SX.sym('beta', 3)
        
        self.g_u = cs.Function('g_u', [self.u_opt, self.s_opt], [cs.vertcat(
            self.miu*self.u_opt[0] - self.u_opt[1],
            self.miu*self.u_opt[0] + self.u_opt[1],
            self.u_opt[4] * (self.miu*self.u_opt[0] - self.u_opt[1]) + self.s_opt[0],
            self.u_opt[5] * (self.miu*self.u_opt[0] + self.u_opt[1]) + self.s_opt[1],
            self.miu*self.u_opt[2] - self.u_opt[3],
        )], ['u', 's'], ['g1'])
        self.lbg1 = [0., 0., 0., 0., 0.]
        self.ubg1 = [cs.inf, cs.inf, 0., 0., 0.]
        
        self.g_xu = cs.Function('g_xu', [self.x_opt, self.u_opt, self.beta], [cs.vertcat(
                                self.vctact(self.x_opt[6]-self.x_opt[2], cs.vertcat(self.x_opt[3], self.x_opt[7]), self.u_opt[0:4], self.beta),
                                self.x_opt[6]-self.x_opt[2])],
                                ['x', 'u', 'b'],
                                ['g2'])
        self.lbg2 = [0., 0., -0.5*np.pi]
        self.ubg2 = [0., self.v_lim, 0.]
        # self.lbg2 = [-cs.inf]*2
        # self.ubg2 = [cs.inf]*2
        
        self.f = cs.Function('f', [self.x_opt, self.u_opt, self.beta],
                             [__f(self.x_opt, self.x_opt[2], self.x_opt[6], self.x_opt[6]-self.x_opt[2], cs.vertcat(self.x_opt[3], self.x_opt[7]), self.u_opt[4]-self.u_opt[5], self.u_opt[0:4], self.beta)],
                             ['x', 'u', 'b'],
                             ['f'])
        
        # slack variable cost
        __i_th = cs.SX.sym('__i_th')
        __Ks_max = self.Ks_max
        __Ks_min = self.Ks_min
        self.ks_f = cs.Function('ks', [__i_th], [__Ks_max * cs.exp(__i_th * cs.log(__Ks_min / __Ks_max))])
        
    def buildOptObj(self, timeHorizon, configDict, X_nom_val, maxIter):
        # init parameters
        # -----------------------
        self.TH = timeHorizon
        self.max_iter = maxIter
        self.solver_name = 'ipopt'
        self.no_printing = configDict['noPrintingFlag']
        
        self.W_x = cs.diag(cs.SX([1.0, 1.0, 0.1, 0., 0., 0., 0., 0.]))
        self.W_u = cs.diag(cs.SX([0.01, 0.01, 0.01, 0.01, 0., 0.]))
        self.K_goal = configDict['K_goal']
        
        self.X_nom_val = X_nom_val
        
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
        self.X = cs.SX.sym('X', self.Nx, self.TH)
        self.U = cs.SX.sym('U', self.Nu, self.TH-1)
        self.S = cs.SX.sym('S', self.Ns, self.TH-1)
        self.X_nom = cs.SX.sym('X_nom', self.Nx, self.TH)
        self.X_bar = self.X - self.X_nom
        
        # initial state
        self.x0 = cs.SX.sym('x0', self.Nx)
        
        # ---- Define Dynamic constraints ----
        __x_next = cs.SX.sym('x_next', self.Nx)
        self.f_error = cs.Function(
                'f_error',
                [self.x_opt, self.u_opt, __x_next, self.beta],
                [__x_next-self.f(self.x_opt,self.u_opt,self.beta)])
        
        self.g1 = cs.Function(
                'g1',
                [self.u_opt, self.s_opt],
                [self.g_u(self.u_opt,self.s_opt)])
        
        self.g2 = cs.Function(
                'g2',
                [self.x_opt,self.u_opt,self.beta],
                [self.g_xu(self.x_opt,self.u_opt,self.beta)])
        
        self.F_error = self.f_error.map(self.TH-1)
        self.G1 = self.g1.map(self.TH-1)
        self.G2 = self.g2.map(self.TH-1)
        
        # --- Define cost functions ----
        __x_bar = cs.SX.sym('x_bar', self.Nx)
        self.cost_f = cs.Function(
                'cost_f',
                [__x_bar, self.u_opt],
                [cs.dot(__x_bar, cs.mtimes(self.W_x, __x_bar)) + 
                 cs.dot(self.u_opt, cs.mtimes(self.W_u, self.u_opt))])
        self.cost_F = self.cost_f.map(self.TH-1)
        
        self.ks_F = self.ks_f.map(self.TH-1)
        index_s = np.linspace(0, 1, self.TH-1)
        self.Ks = self.ks_F(index_s).T
        
        # ---- Set optimization variables ----
        for i in range(self.TH-1):
            # states
            self.opt.x += self.X[:, i].elements()
            if i == 0: # expanding state constraint for 1st one (what trick is this?)
                self.args.lbx += [1.5*x for x in self.lbx]
                self.args.ubx += [1.5*x for x in self.ubx]
            else:
                self.args.lbx += self.lbx
                self.args.ubx += self.ubx
            self.opt.discrete += [False]*self.Nx
            # actions
            self.opt.x += self.U[:, i].elements()
            self.args.lbx += self.lbu
            self.args.ubx += self.ubu
            self.opt.discrete += [False]*self.Nu
        # terminal states
        self.opt.x += self.X[:, -1].elements()
        self.args.lbx += self.lbx
        self.args.ubx += self.ubx
        self.opt.discrete += [False]*self.Nx
        for i in range(self.TH-1):
            # slack variables
            self.opt.x += self.S[:, i].elements()
            self.args.lbx += self.lbs
            self.args.ubx += self.ubs
            self.opt.discrete += [False]*self.Ns
            
        # ---- Set optimization constraints ----
        # initial conditions
        self.opt.g = (self.X[:, 0]-self.x0).elements()
        self.args.lbg = [0.0]*self.Nx
        self.args.ubg = [0.0]*self.Nx
        # dynamic constraints
        self.opt.g += self.F_error(
                self.X[:, :-1], self.U,
                self.X[:, 1:],
                self.beta).elements()
        self.args.lbg += [0.] * self.Nx * (self.TH-1)
        self.args.ubg += [0.] * self.Nx * (self.TH-1)
        
        self.opt.g += self.G1(self.U, self.S).elements()
        self.args.lbg += self.lbg1 * (self.TH-1)
        self.args.ubg += self.ubg1 * (self.TH-1)
        
        self.opt.g += self.G2(self.X[:, :-1], self.U, self.beta).elements()
        self.args.lbg += self.lbg2 * (self.TH-1)
        self.args.ubg += self.ubg2 * (self.TH-1)
        
        # ---- Set optimization cost ----
        # state and input
        self.opt.f = cs.sum2(self.cost_F(self.X_bar[:, :-1], self.U))
        self.opt.f += self.K_goal*self.cost_f(self.X_bar[:, -1], self.U[:, -1])
        # slack variables
        for i in range(self.Ns):
            self.opt.f += cs.sum1(self.Ks*(self.S[i, :].T**2))
            
        # ---- Set optimization parameters ----
        self.opt.p = []
        self.opt.p += self.beta.elements()
        self.opt.p += self.x0.elements()
        self.opt.p += self.X_nom.elements()
        
        # ---- Configure IPOPT solver ----
        opts_dict = {'print_time': 0}
        if self.no_printing:
            opts_dict['ipopt.print_level'] = 0
        opts_dict['ipopt.jac_d_constant'] = 'yes'
        opts_dict['ipopt.warm_start_init_point'] = 'yes'
        opts_dict['ipopt.hessian_constant'] = 'yes'
        opts_dict['ipopt.max_iter'] = self.max_iter
        
        prob = {'f': self.opt.f,
                'x': cs.vertcat(*self.opt.x),
                'g': cs.vertcat(*self.opt.g),
                'p': cs.vertcat(*self.opt.p)}
        
        opts_dict['discrete'] = self.opt.discrete
        self.solver = cs.nlpsol('solver', self.solver_name, prob, opts_dict)
    
    def solveProblem(self, idx, x0, beta):
        # set parameters
        p_ = []
        p_ += beta
        p_ += x0
        p_ += self.X_nom_val[:, idx:(idx+self.TH)].elements()
        
        # set warm start
        self.args.x0 = []
        for i in range(self.TH-1):
            self.args.x0 += self.X_nom_val[:, idx+i].elements()
            self.args.x0 += [0.0]*self.Nu
        self.args.x0 += self.X_nom_val[:, (idx+self.TH)-1].elements()
        
        for i in range(self.TH-1):
            self.args.x0 += self.s0
            
        import pdb; pdb.set_trace()
            
        start_time = time.time()
        sol = self.solver(
                x0=self.args.x0,
                lbx=self.args.lbx, ubx=self.args.ubx,
                lbg=self.args.lbg, ubg=self.args.ubg,
                p=p_)
        t_opt = time.time() - start_time
        resultFlag = self.solver.stats()['success']
        opt_sol = sol['x']
        f_opt = sol['f']
        
        self.Nxu = self.Nx + self.Nu
        opt_sol_xu = opt_sol[:(self.TH*self.Nxu-self.Nu)]
        opt_sol_s = opt_sol[(self.TH*self.Nxu-self.Nu):]
        
        x_opt = []
        for i in range(self.Nx):
            x_opt = cs.vertcat(x_opt, opt_sol_xu[i::self.Nxu].T)
        u_opt = []
        for i in range(self.Nx, self.Nxu):
            u_opt = cs.vertcat(u_opt, opt_sol_xu[i::self.Nxu].T)
        s_opt = []
        for i in range(self.Ns):
            s_opt = cs.vertcat(s_opt, opt_sol_s[i::self.Ns].T)
        
        return resultFlag, x_opt, u_opt, s_opt, f_opt, t_opt
    
    
if __name__ == '__main__':
    config = sliding_pack.load_config('tracking_config_2sliders.yaml')
    dyn = Double_Slider_lim_surf_test_dyn(config['dynamics'])
    
    beta = [0.07, 0.12, 0.01]
    T = 5.0
    N = 150
    N_MPC = 0
    
    ya0, yb0 = [0., 0.25*beta[1]]
    x_a0 = [0., 0., 0.5*np.pi]
    x_b0 = [0., 0., 0.25*np.pi]
    x_b0[:2] = (np.array(x_a0[:2]) + sliding_pack.db_sim.rotation_matrix2X2(x_a0[2]) @ np.array([0.5*beta[0], -0.5*beta[1]]) \
                                   - sliding_pack.db_sim.rotation_matrix2X2(x_b0[2]) @ np.array([-0.5*beta[0], yb0])).tolist()
    
    x_init = np.r_[x_a0, ya0, x_b0, yb0].tolist()
    x_goal = [0.05, 0.4, 0.5*np.pi]
    x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(x_goal[0], x_goal[1], N, N_MPC)
    x_nom, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt=0.04)
    x_nom = cs.vertcat(x_nom, cs.DM.zeros(4, N))
    
    dyn.buildOptObj(N, config['TO'], x_nom, maxIter=800)
    resultFlag, x_opt, u_opt, s_opt, f_opt, t_opt = dyn.solveProblem(0, x_init, beta)
    
    np.save('./data/x_opt.npy', x_opt)
    np.save('./data/u_opt.npy', u_opt)
    np.save('./data/s_opt.npy', s_opt)
    
    import pdb; pdb.set_trace()
    
    x_opt = np.load('./data/x_opt.npy')
    u_opt = np.load('./data/u_opt.npy')
    s_opt = np.load('./data/s_opt.npy')
    
    fig, axs = plt.subplots(3, 4, sharex=True)
    fig.set_size_inches(10, 10, forward=True)
    t_Nx = np.linspace(0, T, N)
    t_Nu = np.linspace(0, T, N-1)
    
    for i in [0, 1, 2, 3]:
        axs[0, 0].plot(t_Nx, x_opt[i, 0:N].T, linestyle='--', label='x{0}'.format(i))
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(handles, labels)
    axs[0, 0].set_xlabel('time [s]')
    axs[0, 0].set_ylabel('y [m]')
    axs[0, 0].grid()
    
    for i in [4, 5, 6, 7]:
        axs[0, 1].plot(t_Nx, x_opt[i, 0:N].T, linestyle='--', label='x{0}'.format(i))
    handles, labels = axs[0, 1].get_legend_handles_labels()
    axs[0, 1].legend(handles, labels)
    axs[0, 1].set_xlabel('time [s]')
    axs[0, 1].set_ylabel('y [m]')
    axs[0, 1].grid()
    
    for i in [0, 1]:
        axs[1, 0].plot(t_Nu, u_opt[i, 0:N].T, linestyle='--', label='x{0}'.format(i))
    axs[1, 0].plot(t_Nu, (u_opt[4, 0:N] - u_opt[5, 0:N]).T, linestyle='--', label='x{0}'.format(i))
    handles, labels = axs[1, 0].get_legend_handles_labels()
    axs[1, 0].legend(handles, labels)
    axs[1, 0].set_xlabel('time [s]')
    axs[1, 0].set_ylabel('y [m]')
    axs[1, 0].grid()
    
    for i in [2, 3]:
        axs[1, 1].plot(t_Nu, u_opt[i, 0:N].T, linestyle='--', label='x{0}'.format(i))
    handles, labels = axs[1, 1].get_legend_handles_labels()
    axs[1, 1].legend(handles, labels)
    axs[1, 1].set_xlabel('time [s]')
    axs[1, 1].set_ylabel('y [m]')
    axs[1, 1].grid()
    
    for i in [0, 1]:
        axs[2, 0].plot(t_Nu, s_opt[i, 0:N].T, linestyle='--', label='x{0}'.format(i))
    handles, labels = axs[2, 0].get_legend_handles_labels()
    axs[2, 0].legend(handles, labels)
    axs[2, 0].set_xlabel('time [s]')
    axs[2, 0].set_ylabel('y [m]')
    axs[2, 0].grid()
    
    import pdb; pdb.set_trace()
    
    ctrl_g_u_idx = dyn.g_u.map(N-1)
    ctrl_g_u_val = ctrl_g_u_idx(u_opt, s_opt)
    
    ctrl_g_xu_idx = dyn.g_xu.map(N-1)
    ctrl_g_xu_val = ctrl_g_xu_idx(x_opt[:, :-1], u_opt, beta)
    
    dyn_f_error_val = dyn.F_error(x_opt[:, :-1], u_opt, x_opt[:, 1:], beta)
    
    for i in [0, 1, 2, 3, 4]:
        axs[0, 2].plot(t_Nu, ctrl_g_u_val[i, 0:N-1].T, linestyle='--', label='x{0}'.format(i))
    handles, labels = axs[0, 2].get_legend_handles_labels()
    axs[0, 2].legend(handles, labels)
    axs[0, 2].set_xlabel('time [s]')
    axs[0, 2].set_ylabel('y [m]')
    axs[0, 2].grid()
    
    for i in [0, 1, 2]:
        axs[0, 3].plot(t_Nu, ctrl_g_xu_val[i, 0:N-1].T, linestyle='--', label='x{0}'.format(i))
    handles, labels = axs[0, 3].get_legend_handles_labels()
    axs[0, 3].legend(handles, labels)
    axs[0, 3].set_xlabel('time [s]')
    axs[0, 3].set_ylabel('y [m]')
    axs[0, 3].grid()
    
    for i in [0, 1, 2, 3]:
        axs[1, 2].plot(t_Nu, dyn_f_error_val[i, 0:N-1].T, linestyle='--', label='x{0}'.format(i))
    handles, labels = axs[1, 2].get_legend_handles_labels()
    axs[1, 2].legend(handles, labels)
    axs[1, 2].set_xlabel('time [s]')
    axs[1, 2].set_ylabel('y [m]')
    axs[1, 2].grid()
    
    for i in [4, 5, 6, 7]:
        axs[1, 3].plot(t_Nu, dyn_f_error_val[i, 0:N-1].T, linestyle='--', label='x{0}'.format(i))
    handles, labels = axs[1, 3].get_legend_handles_labels()
    axs[1, 3].legend(handles, labels)
    axs[1, 3].set_xlabel('time [s]')
    axs[1, 3].set_ylabel('y [m]')
    axs[1, 3].grid()
    
    plt.show()
    
    import pdb; pdb.set_trace()
    
    simObj = sliding_pack.db_sim.buildDoubleSliderSimObj(
        N, config['dynamics'], 0.04, method='qpoases',
        showAnimFlag=True, saveFileFlag=False)
    
    simObj.visualization(x_opt[0:3, :].T, x_opt[4:7, :].T, np.c_[-np.ones(N)*0.5*beta[0], x_opt[3, :]], beta)
        