# Author: Yongpeng Jiang
# Contact: jyp19@mails.tsinghua.edu.cn
# Date: 11/22/2022
# -------------------------------------------------------------------
# Description:
# 
# Class for the trajectory optimization (TO) for the pusher-slider 
# problem using a Differential Dynamic Program (DDP) approach
# -------------------------------------------------------------------

# import libraries
import sys
import os
import time
import numpy as np
import casadi as cs
import cvxopt as cvx
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

class buildDDPOptObj():
    def __init__(self, dyn_class, timeHorizon, configDict) -> None:
        """
            configDict: the key ['TO'] of .yaml file
        """
        # init parameters
        self.dyn = dyn_class
        self.TH = timeHorizon
        self.miu = dyn_class.miu
        self.beta = dyn_class.beta
        
        self.W_x = cs.diag(cs.SX(configDict['W_x']))
        self.W_u = cs.diag(cs.SX(configDict['W_u'][:3]))
        self.gamma_u = cs.diag(cs.SX(configDict['gamma_u']))
        
        # input constraints
        self.A_u = np.array([[ 1, 0, 0],
                             [ 0, 1, 0],
                             [ 0, 0, 1],
                             [self.miu,-1, 0],
                             [self.miu, 1, 0]])

        self.lb_u = np.array([[0, -self.dyn.f_lim, -self.dyn.psi_dot_lim, 0, 0]])

        self.ub_u = np.array([[self.dyn.f_lim, self.dyn.f_lim, self.dyn.psi_dot_lim, np.inf, np.inf]])
        
        # dynamic functions
        self.f_xu = cs.Function(
            'f_xu',
            [self.dyn.x, self.dyn.u, self.dyn.beta],
            [self.dyn.f_(self.dyn.x, self.dyn.u, self.dyn.beta)],
            ['x', 'u', 'b'],
            ['f_xu']
        )
        
        self.build_value_approx()
        
    def build_value_approx(self):
        """
            build the symbol of quadratic approximation matrix Q
            with Casadi
        """
        # auxiliar symbolic variables
        # -------------------------------------------------------------------
        # dx - state vector
        __dx = cs.SX.sym('x', 4)
        # du - control vector
        __du = cs.SX.sym('u', 3)
        # [1, dx, du] - concat input vector
        __dxu = cs.SX(8, 1)
        __dxu[0, 0] = 1
        __dxu[1:5, 0] = __dx
        __dxu[5:, 0] = __du
        # nominal state
        __nom_x = cs.SX.sym('nom_x', 4)
        
        #  -------------------------------------------------------------------
        __cost_l = cs.Function(
            'cost_l',
            [self.dyn.x, self.dyn.u, __nom_x],
            [cs.dot((self.dyn.x - __nom_x), cs.mtimes(self.W_x, (self.dyn.x - __nom_x)))
             + cs.dot(self.dyn.u, cs.mtimes(self.W_u, self.dyn.u))]
        )
        
        __cost_VN = cs.dot((self.dyn.x - __nom_x), cs.mtimes(self.W_x, (self.dyn.x - __nom_x)))

        # first and second derivatives of V_{k+1}, l and f
        __lx = cs.gradient(__cost_l(self.dyn.x, self.dyn.u, __nom_x), self.dyn.x)
        __lu = cs.gradient(__cost_l(self.dyn.x, self.dyn.u, __nom_x), self.dyn.u)
        __lxx = cs.hessian(__cost_l(self.dyn.x, self.dyn.u, __nom_x), self.dyn.x)[0]
        __luu = cs.hessian(__cost_l(self.dyn.x, self.dyn.u, __nom_x), self.dyn.u)[0]
        __lux = cs.jacobian(cs.gradient(__cost_l(self.dyn.x, self.dyn.u, __nom_x), self.dyn.u), self.dyn.x)
        __fx = cs.jacobian(self.f_xu(self.dyn.x, self.dyn.u, self.beta), self.dyn.x)
        __fu = cs.jacobian(self.f_xu(self.dyn.x, self.dyn.u, self.beta), self.dyn.u)
        __fxx = [cs.hessian(self.f_xu(self.dyn.x, self.dyn.u, self.beta)[i], self.dyn.x)[0] for i in range(4)]
        __fuu = [cs.hessian(self.f_xu(self.dyn.x, self.dyn.u, self.beta)[i], self.dyn.u)[0] for i in range(4)]
        __fux = [cs.jacobian(cs.gradient(self.f_xu(self.dyn.x, self.dyn.u, self.dyn.beta)[i], self.dyn.u), self.dyn.x) for i in range(4)]

        self.VN = cs.Function(
            'VN',
            [self.dyn.x, __nom_x],
            [__cost_VN],
            ['x', 'nom_x'],
            ['VN']
        )

        __Vx = cs.SX.sym('_Vx', 4, 1)
        __Vxx = cs.SX.sym('_Vxx', 4, 4)
        __VNx = cs.gradient(self.VN(self.dyn.x, __nom_x), self.dyn.x)
        __VNxx = cs.hessian(self.VN(self.dyn.x, __nom_x), self.dyn.x)[0]

        self.VNx = cs.Function(
            'VNx',
            [self.dyn.x, __nom_x],
            [__VNx],
            ['x', 'nom_x'],
            ['VNx']
        )

        self.VNxx = cs.Function(
            'VNxx',
            [self.dyn.x, __nom_x],
            [__VNxx],
            ['x', 'nom_x'],
            ['VNxx']
        )
        
        __qx = __lx + cs.mtimes(__fx.T, __Vx)
        __qu = __lu + cs.mtimes(__fu.T, __Vx)
        __Qxx = __lxx + cs.mtimes(__fx.T, cs.mtimes(__Vxx, __fx)) + \
                sum((__Vx[i] * __fxx[i]) for i in range(4))
        __Quu = __luu + cs.mtimes(__fu.T, cs.mtimes(__Vxx, __fu)) + \
                sum((__Vx[i] * __fuu[i]) for i in range(4))
        __Qux = __lux + cs.mtimes(__fu.T, cs.mtimes(__Vxx, __fx)) + \
                sum((__Vx[i] * __fux[i]) for i in range(4))
        
        self.Q_xx = cs.Function(
            'Q_xx',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__Qxx]
        )

        self.Q_uu = cs.Function(
            'Q_uu',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__Quu]
        )
        
        self.Q_ux = cs.Function(
            'Q_ux',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__Qux]
        )
        
        self.qx = cs.Function(
            'qx',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__lx + cs.mtimes(__fx.T, __Vx)]
        )

        self.qu = cs.Function(
            'qu',
            [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
            [__qu]
        )
        
        # self.fx = cs.Function(
        #     'fu',
        #     [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
        #     [__fx]
        # )
        
        # self.lx = cs.Function(
        #     'lu',
        #     [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
        #     [__lx]
        # )
        
        # self.Vxx = cs.Function(
        #     'Vxx',
        #     [self.dyn.x, self.dyn.u, self.dyn.beta, __nom_x, __Vx, __Vxx],
        #     [__Vxx]
        # )
        
        # test code
        xx = [1.1, 2.2, 3.3, 4.4]
        uu = [5, 6, 7]
        nom_x = [1, 2, 3, 4]
        beta = [0.07, 0.12, 0.01]
        Vx_vec = np.random.rand(4, 1)
        Vxx_mat = np.random.rand(4, 4)
        var = xx, uu, beta, nom_x, Vx_vec, Vxx_mat
        
        import pdb; pdb.set_trace()
        
    def make_quadratic_approx(self):
        """
            make useful blocks storage of the matrix Q
            ------output
            Dict[List[]]
        """
        self.Q_block = {
            'Q_xx': [],  # DM(4, 4)
            'Q_uu': [],  # DM(3, 3)
            'Q_ux': [],  # DM(3, 4)
            'qx': [],  # DM(4, 1)
            'qu': [],  # DM(3, 1)
        }
        
    def project_feasible(self, k, uk, duk, dx):
        """
            get the inputs that do not violate constraints
            in the forward propagate process
        """
        Q_ux = self.Q_block['Q_ux'][k]
        Q_uu = self.Q_block['Q_uu'][k]
        qu = self.Q_block['qu'][k]
        
        H = Q_uu + self.gamma_u
        G = Q_ux @ dx + qu - self.gamma_u.T @ duk
        
        A_u = cs.DM(self.A_u)  # DM(3, 3)

        lb_a = self.lb_u - self.A_u @ uk
        ub_a = self.ub_u - self.A_u @ uk
        
        qp = {'h': H.sparsity(),
              'a': A_u.sparsity()}
        S = cs.conic('S', 'qpoases', qp)
        r = S(h=H, \
              g=G, \
              a=A_u, \
              lba = lb_a, \
              uba = ub_a)
        duk_hat = r['x']  # suppose (3, 1)
        
        return uk + duk_hat
        
    def coldboot_integration(self, x0, U0):
        """
            ------ input
            x0: (4, 1) is the array of initial states
            U0: (3, N) is the array of initial inputs
            ------ output
            X_hat: (4, N+1) is the array of updated states
        """
        X_hat = np.zeros((4, self.TH + 1))
        X_hat[:, 0] = x0  # (4, 1)
        for k in range(0, self.TH):
            X_hat[:, k+1] = X_hat[:, k] + self.f_xu(X_hat[:, k], U0[:, k], self.beta)

        return X_hat

    def backward_propagation(self, X, U):
        """
            ------ input
            X: (4, N+1) is the array of forward integrated states
            U: (4, N) is the array of inputs
            ------ output
            k_vec: (N, 3) is the set of feedforward vectors
            K_mat: (N, 3, 4) is the set of feedback matrices
        """
        self.make_quadratic_approx()
        
        VN = self.VN([X[:, -1]]).toarray()[0, 0]  # double
        V = VN  # value function
        Vx, Vxx = np.zeros((4, 1)), np.zeros((4, 4))
        self.Vx_tensor, self.Vxx_tensor = np.zeros((self.TH, 4, 1)), np.zeros((self.TH, 4, 4))
        self.k_vec, self.K_mat = np.zeros((self.TH, 3, 1)), np.zeros((self.TH, 3, 4))
        
        for k in range(self.TH - 1, -1, -1):
            if k == self.TH - 1:
                Vx = self.VNx([X[:, -1]]).toarray()  # (4, 1)
                Vxx = self.VNxx([X[:, -1]]).toarray()  # (4, 4)

            # calculate matrix values
            Q_xx = self.Q_xx(X[:, k], U[:, k], self.beta, self.X_nom[:, k], Vx, Vxx)
            Q_ux = self.Q_ux(X[:, k], U[:, k], self.beta, self.X_nom[:, k], Vx, Vxx)
            Q_uu = self.Q_uu(X[:, k], U[:, k], self.beta, self.X_nom[:, k], Vx, Vxx)
            qx = self.qx(X[:, k], U[:, k], self.beta, self.X_nom[:, k], Vx, Vxx)
            qu = self.qu(X[:, k], U[:, k], self.beta, self.X_nom[:, k], Vx, Vxx)
            
            self.Q_block['Q_xx'].insert(0, Q_xx)
            self.Q_block['Q_ux'].insert(0, Q_ux)
            self.Q_block['Q_uu'].insert(0, Q_uu)
            self.Q_block['qx'].insert(0, qx)
            self.Q_block['qu'].insert(0, qu)

            A_u = cs.DM(self.A_u)  # DM(3, 3)

            lb_a = self.lb_u - self.A_u @ U[:, k]
            ub_a = self.ub_u - self.A_u @ U[:, k]

            # solve the qp problem for feedforward k
            qp = {'h': Q_uu.sparsity(),
                  'a': A_u.sparsity()}
            S = cs.conic('S', 'qpoases', qp)
            r = S(h=Q_uu, \
                  g=qu, \
                  a=A_u, \
                  lba = lb_a, \
                  uba = ub_a)
            du = r['x']  # suppose (3, 1)

            # solve for feedback matrices K
            eps = 1e-8
            grad = qu.toarray() + Q_uu.toarray() @ du
            c_x = np.where(np.bitwise_or(np.bitwise_and(np.abs(self.A_u @ du - lb_a) < eps, self.A_u @ grad > 0), \
                                         np.bitwise_and(np.abs(self.A_u @ du - ub_a) < eps, self.A_u @ grad < 0)) == 1)
            invQ_uu = np.linalg.inv(Q_uu.toarray())
            invQ_uu[c_x, :] = 0
            
            self.k_vec[k, :] = np.array(du)
            self.K_mat[k, ...] = -invQ_uu @ Q_ux.toarray()
            
            # propagate value
            V += 0.5 * self.k_vec[k, :] @ Q_uu.toarray() @ self.k_vec[k, :]
            Vx = qx - self.K_mat[k, ...].T @ Q_uu.toarray() @ self.k_vec[k, :]
            Vxx = Q_xx - self.K_mat[k, ...].T @ Q_uu.toarray() @ self.K_mat[k, ...]
            self.Vx_tensor[k, :], self.Vxx_tensor[k, ...] = Vx, Vxx

    def forward_integration(self, X, U):
        """
            X: (4, N+1) is the array of current states
            U: (3, N) is the array of current inputs
            k: (3, 1) is the feedforward vector
            K: (3, 4) is the feedback matrix
        """
        X_hat, U_hat = np.zeros((4, self.TH + 1)), np.zeros((3, self.TH))
        X_hat[:, 0] = X[:, 0]  # (4, 1)
        for k in range(0, self.TH):
            k_vec, K_mat = self.k_vec[k, :], self.K_mat[k, ...]
            dx = X_hat[:, k] - X[:, k]
            U_hat[:, k] = self.project_feasible(k, U[:, k], k_vec + K_mat @ dx, dx)  # (3, 1)
            X_hat[:, k+1] = X_hat[:, k] + self.f_xu(X_hat[:, k], U_hat[:, k]).toarray()
        
        return X_hat, U_hat
        
    def solve_constrained_ddp(self, x_init, U_init):
        """
            x_init: (4, 1)
            U_init: (3, N)
            beta: list[3]
        """
        
if __name__ == '__main__':
    planning_config = sliding_pack.load_config('planning_switch_config.yaml')
    dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
        planning_config['dynamics'],
        planning_config['TO']['contactMode']
    )
    T = 2.5  # time of the simulation is seconds
    freq = 25  # number of increments per second
    dt = 1.0/freq  # sampling time
    N = int(T*freq)  # total number of iterations
    ddpOptObj = buildDDPOptObj(dyn_class=dyn,
                               timeHorizon=N,
                               configDict=planning_config['TO'])
    