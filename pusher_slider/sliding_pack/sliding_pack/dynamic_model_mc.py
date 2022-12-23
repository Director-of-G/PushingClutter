# Author: Joao Moura (Modifed by Yongpeng Jiang)
# Contact: jpousad@ed.ac.uk (jyp19@mails.tsinghua.edu.cn)
# Date: 19/10/2020 (Modified on 17/12/2022)
# -------------------------------------------------------------------
# Description:
# 
# Functions modelling the dynamics of an object sliding on a table.
# Based on: Jiaji Zhou, J. Andrew Bagnell and Matthew T. Mason
#           2017 paper (arxiv: https://arxiv.org/pdf/1705.10664.pdf)
# Modified: from dynamic_mode.py in sliding_pack
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------
import sys
import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import casadi as cs
# -------------------------------------------------------------------
import sliding_pack
# -------------------------------------------------------------------

class Double_Sys_sq_slider_quasi_static_ellip_lim_surf():
    # The dynamic model for single-pusher-double-slider.
    # The pusher directly manipulate slider A. And slider A pushes slider B.
    # The slider A has a face contact. and the slider B has a vertex contact.
    # The dynamic model is under quasi-static assumption.
    # The dynamic model is approximated by an ellipsoid.
    def __init__(self, configDict, contactNum=2., contactMode='A_face_B_edge', controlRelPose=True):

        # init parameters
        #  -------------------------------------------------------------------
        self.miu = configDict['pusherFricCoef']  # fric between pusher and slider
        self.f_lim = configDict['pusherForceLim']
        self.v_lim = configDict['pusherVelLim']
        self.x_length = configDict['xLenght']
        self.y_length = configDict['yLenght']
        self.Kx_max = configDict['Kx_max']
        self.Ks_max = configDict['Ks_max']
        self.Ks_min = configDict['Ks_min']
        self.n_ctact = contactNum  # number of contacts (including the pusher)
        self.contactMode = contactMode  # 'A_face_B_edge' or 'A_edge_B_face
        self.controlRelPose = controlRelPose  # if true, only relative pose will be considered; if false, we control sliderA pose instead.
        #  -------------------------------------------------------------------
        self.Nbeta = 3
        self.beta = cs.SX.sym('beta', self.Nbeta)
        self.theta = cs.SX.sym('theta', 2)
        self.ctact = cs.SX.sym('ctact', 3, 2)
        self.vp = cs.SX.sym('vp', 2)
        #  -------------------------------------------------------------------

        # variables
        # -------------------------------------------------------------------
        # contact force
        __f_norm = cs.SX.sym('__f_norm', self.n_ctact)
        __f_tan = cs.SX.sym('__f_tan', 2*self.n_ctact)
        __f_ctact = cs.vertcat(__f_norm, __f_tan)
        # -------------------------------------------------------------------
        # auxiliary
        __alpha = cs.SX.sym('__alpha', self.n_ctact)
        __beta = cs.SX.sym('__beta', 2*self.n_ctact)
        __lamda = cs.SX.sym('__lamda', self.n_ctact)
        __gamma = cs.SX.sym('__gamma', self.n_ctact)
        __aux = cs.vertcat(__alpha, __beta, __lamda, __gamma)
        # -------------------------------------------------------------------
        # pusher velicity
        __vpx = cs.SX.sym('__vpx')
        __vpy = cs.SX.sym('__vpy')
        __vp = cs.vertcat(__vpx, __vpy)
        # -------------------------------------------------------------------
        # stack vector
        self.w = cs.vertcat(__alpha, __beta, __gamma)
        self.z = cs.vertcat(__f_ctact, __lamda)
        # -------------------------------------------------------------------
        
        # system model
        # -------------------------------------------------------------------
        # geometry
        __xl = cs.SX.sym('__xl')  # slider x lenght
        __yl = cs.SX.sym('__yl')  # slider y lenght
        __r_pusher = cs.SX.sym('__r_pusher')  # radious of the cilindrical pusher
        __Beta = cs.veccat(__xl, __yl, __r_pusher)
        # limit surface
        __Area = __xl*__yl
        __int_Area = sliding_pack.integral.rect_cs(__xl, __yl)
        __c = __int_Area/__Area  # ellipsoid approximation ratio
        __A = cs.SX.sym('__A', cs.Sparsity.diag(3))
        __A[0,0] = __A[1,1] = 1.; __A[2,2] = 1./(__c**2)
        self.A = cs.Function('A', [__Beta], [__A], ['b'], ['A'])
        # -------------------------------------------------------------------
        # theta - state variables
        __thetaA = cs.SX.sym('__thetaA')
        __thetaB = cs.SX.sym('__thetaB')
        __theta = cs.vertcat(__thetaA, __thetaB)
        __dtheta = __thetaB - __thetaA
        # R - rotation matrices
        __RA, __RB = cs.SX(3, 3), cs.SX(3, 3)
        __cthetaA = cs.cos(__thetaA); __sthetaA = cs.sin(__thetaA)
        __RA[0,0] = __cthetaA; __RA[0,1] = -__sthetaA; __RA[1,0] = __sthetaA; __RA[1,1] = __cthetaA; __RA[2,2] = 1.0
        __cthetaB = cs.cos(__thetaB); __sthetaB = cs.sin(__thetaB)
        __RB[0,0] = __cthetaB; __RB[0,1] = -__sthetaB; __RB[1,0] = __sthetaB; __RB[1,1] = __cthetaB; __RB[2,2] = 1.0
        self.RA = cs.Function('RA', [__thetaA], [__RA], ['t'], ['ra'])
        self.RB = cs.Function('RB', [__thetaB], [__RB], ['t'], ['rb'])
        # T - planar transformation matrices
        # __TB2A = cs.mtimes(cs.inv(__RA[:2,:2]), __RB[:2,:2]); __TA2B = cs.mtimes(cs.inv(__RB[:2,:2]), __RA[:2,:2])
        __TB2A, __TA2B = cs.SX(2, 2), cs.SX(2, 2)
        __cdtheta = cs.cos(__dtheta); __sdtheta = cs.sin(__dtheta)
        __TB2A[0,0] = __cdtheta; __TB2A[0,1] = -__sdtheta; __TB2A[1,0] = __sdtheta; __TB2A[1,1] = __cdtheta
        __TA2B[0,0] = __cdtheta; __TA2B[0,1] = __sdtheta; __TA2B[1,0] = -__sdtheta; __TA2B[1,1] = __cdtheta
        # -------------------------------------------------------------------

        # contact jacobian
        #  -------------------------------------------------------------------
        # contact locations
        # __xA0 = cs.SX.sym('__xA0'); __yA0 = cs.SX.sym('__yA0')
        # __xA1 = cs.SX.sym('__xA1'); __yA1 = cs.SX.sym('__yA1')
        # __xB0 = cs.SX.sym('__xB0'); __yB0 = cs.SX.sym('__yB0')
        __xA0 = -__Beta[0]/2; __yA0 = cs.SX.sym('__yA0')
        __xA1 = __Beta[0]/2; __yA1 = cs.SX.sym('__yA1')
        __xB0 = -__Beta[0]/2; __yB0 = -__Beta[1]/2
        __ctact = cs.SX(3, 2)
        __ctact[0, 0] = __xA0; __ctact[0, 1] = __yA0
        __ctact[1, 0] = __xA1; __ctact[1, 1] = __yA1
        __ctact[2, 0] = __xB0; __ctact[2, 1] = __yB0
        # contact jacobian
        __JA0, __JA1, __JB0 = cs.SX(2, 3), cs.SX(2, 3), cs.SX(2, 3)
        __JA0[0, 0] = 1; __JA0[1, 1] = 1; __JA0[0, 2] = -__yA0; __JA0[1, 2] = __xA0
        __JA1[0, 0] = 1; __JA1[1, 1] = 1; __JA1[0, 2] = -__yA1; __JA1[1, 2] = __xA1
        __JB0[0, 0] = 1; __JB0[1, 1] = 1; __JB0[0, 2] = -__yB0; __JB0[1, 2] = __xB0
        #  -------------------------------------------------------------------

        # stacking matrix
        #  -------------------------------------------------------------------
        # auxiliary matrix
        __K = cs.mtimes(cs.mtimes(cs.mtimes(__TB2A, __JB0), __A), cs.mtimes(__JB0.T, __TA2B))
        # self.K = cs.Function('K', [__Beta, __ctact, __theta], [__K], ['b', 'c', 't'], ['K'])
        #  -------------------------------------------------------------------
        # normal vectors
        __nA0 = np.array([[1., 0.]])
        __nA1 = np.array([[-1., 0.]])
        __nB0 = np.array([[1., 0.]])
        __N = cs.vertcat(cs.mtimes(__nA0, __JA0), cs.mtimes(__nA1, __JA1))
        __Nv = cs.vertcat(cs.mtimes(__nA0, __JA0), cs.mtimes(__nA1, __JA1))
        # tangent vectors
        __DA0 = np.array([[0., 1.],
                          [0., -1.]]).T
        __DA1 = np.array([[0., -1.],
                          [0., 1.]]).T
        __DB0 = np.array([[0., 1.],
                          [0., -1.]]).T
        __L = cs.vertcat(cs.mtimes(__DA0.T, __JA0), cs.mtimes(__DA1.T, __JA1))
        __Lv = cs.vertcat(cs.mtimes(__DA0.T, __JA0), cs.mtimes(__DA1.T, __JA1))
        #  -------------------------------------------------------------------
        # sa, sb represented with fn, ft, the coefficient matrices are
        #  -------------------------------------------------------------------
        # selection matrix
        __nsel = np.array([[0, 1],
                           [0, 0]])
        __tsel = np.array([[0, 0, 0, 0],
                           [0, 0, 1, -1]])
        __N1 = cs.SX(2, 2); __N1[0, :] = 0; __N1[1, :] = -cs.mtimes(__nA1, cs.mtimes(__K, __nsel))
        __L1 = cs.SX(2, 4); __L1[0, :] = 0; __L1[1, :] = -cs.mtimes(__nA1, cs.mtimes(__K, __tsel))
        __N2 = cs.SX(4, 2); __N2[:2, :] = 0; __N2[2:, :] = -cs.mtimes(__DA1.T, cs.mtimes(__K, __nsel))
        __L2 = cs.SX(4, 4); __L2[:2, :] = 0; __L2[2:, :] = -cs.mtimes(__DA1.T, cs.mtimes(__K, __tsel))
        #  -------------------------------------------------------------------
        __E = np.kron(np.eye(2), np.ones((2, 1)))
        __mu = self.miu * np.eye(2)
        # __mu = np.diag([0.2, 0.5])
        #  -------------------------------------------------------------------

        # lcp problems (wiki: https://en.wikipedia.org/wiki/Linear_complementarity_problem#cite_note-FOOTNOTEFukudaNamiki1994-6)
        #  -------------------------------------------------------------------
        __M = cs.SX(4*self.n_ctact, 4*self.n_ctact)

        __M[:self.n_ctact, :self.n_ctact] = cs.mtimes(__Nv, cs.mtimes(__A, __N.T)) + __N1
        __M[:self.n_ctact, self.n_ctact: 3*self.n_ctact] = cs.mtimes(__Nv, cs.mtimes(__A, __L.T)) + __L1
        __M[:self.n_ctact, 3*self.n_ctact:] = 0

        __M[self.n_ctact: 3*self.n_ctact, :self.n_ctact] = cs.mtimes(__Lv, cs.mtimes(__A, __N.T)) + __N2
        __M[self.n_ctact: 3*self.n_ctact, self.n_ctact: 3*self.n_ctact] = cs.mtimes(__Lv, cs.mtimes(__A, __L.T)) + __L2
        __M[self.n_ctact: 3*self.n_ctact, 3*self.n_ctact:] = __E

        __M[3*self.n_ctact:, :self.n_ctact] = __mu
        __M[3*self.n_ctact:, self.n_ctact: 3*self.n_ctact] = -__E.T
        __M[3*self.n_ctact:, 3*self.n_ctact:] = 0
        #  -------------------------------------------------------------------
        __sa = cs.vertcat(-cs.mtimes(__nA0, __vp), 0)
        __sb = cs.vertcat(-cs.mtimes(__DA0.T, __vp), 0, 0)
        __q = cs.vertcat(__sa, __sb, 0, 0)
        #  -------------------------------------------------------------------
        self.M_ = cs.Function('M_', [__Beta, __theta, __yA0, __yA1],
                              [__M], ['b', 't', 'ya0', 'ya1'], ['M_'])
        self.M_opt_ = self.M_
        self.M = cs.Function('M', [self.beta, self.theta, self.ctact],
                             [self.M_(self.beta, self.theta, self.ctact[0,1], self.ctact[1,1])], ['b', 't', 'c'], ['M'])
        self.q_ = cs.Function('q_', [__vp], [__q], ['v'], ['q'])
        self.q_opt_ = self.q_
        self.q = cs.Function('q', [self.vp], [self.q_(self.vp)], ['v'], ['q'])
        #  -------------------------------------------------------------------

        # dynamic functions
        #  -------------------------------------------------------------------
        # sliderA, sliderB wrench
        __wrenchA = cs.mtimes(__N.T, __f_norm) + cs.mtimes(__L.T, __f_tan)
        __wrenchB = cs.mtimes(__JB0.T,cs.mtimes(__TA2B,cs.mtimes(__nsel,__f_norm)+cs.mtimes(__tsel,__f_tan)))
        self.wrenchA_ = cs.Function('wrenchA_', [__Beta,__theta,__yA0,__yA1,__f_ctact], [__wrenchA], ['b', 't', 'ya0', 'ya1', 'f'], ['wA_'])
        self.wrenchB_ = cs.Function('wrenchB_', [__Beta,__theta,__yA0,__yA1,__f_ctact], [__wrenchB], ['b', 't', 'ya0', 'ya1', 'f'], ['wB_'])
        self.wrenchA = cs.Function('wrenchA', [self.beta,self.theta,self.ctact,self.z],
                                   [self.wrenchA_(self.beta,self.theta,self.ctact[0,1],self.ctact[1,1],self.z[:6])], ['b', 't', 'c', 'z'], ['wA'])
        self.wrenchB = cs.Function('wrenchB', [self.beta,self.theta,self.ctact,self.z],
                                   [self.wrenchB_(self.beta,self.theta,self.ctact[0,1],self.ctact[1,1],self.z[:6])], ['b', 't', 'c', 'z'], ['wB'])
        # sliderA twist
        __VA = cs.mtimes(__A, __wrenchA)
        __VB = cs.mtimes(__A, __wrenchB)
        self.VA_ = cs.Function('VA_', [__Beta,__theta,__yA0,__yA1,__f_ctact], [__VA], ['b', 't', 'ya0', 'ya1', 'f'], ['VA_'])
        self.VB_ = cs.Function('VB_', [__Beta,__theta,__yA0,__yA1,__f_ctact], [__VB], ['b', 't', 'ya0', 'ya1', 'f'], ['VB_'])
        self.VA = cs.Function('VA', [self.beta,self.theta,self.ctact,self.z],
                              [self.VA_(self.beta,self.theta,self.ctact[0,1],self.ctact[1,1],self.z[:6])], ['b', 't', 'c', 'z'], ['VA'])
        self.VB = cs.Function('VB', [self.beta,self.theta,self.ctact,self.z],
                              [self.VB_(self.beta,self.theta,self.ctact[0,1],self.ctact[1,1],self.z[:6])], ['b', 't', 'c', 'z'], ['VB'])
        # slider pose
        __fA = cs.mtimes(__RA,__VA)
        __fB = cs.mtimes(__RB,__VB)
        self.fA_ = cs.Function('fA_', [__Beta,__theta,__yA0,__yA1,__f_ctact], [__fA], ['b', 't', 'ya0', 'ya1', 'c'], ['fa_'])
        self.fB_ = cs.Function('fB_', [__Beta,__theta,__yA0,__yA1,__f_ctact], [__fB], ['b', 't', 'ya0', 'ya1', 'c'], ['fb_'])
        self.fA = cs.Function('fA', [self.beta, self.theta, self.ctact, self.z],
                            [self.fA_(self.beta,self.theta,self.ctact[0,1],self.ctact[1,1],self.z[:6])], ['b', 't', 'c', 'z'], ['fa'])
        self.fB = cs.Function('fB', [self.beta, self.theta, self.ctact, self.z],
                            [self.fB_(self.beta,self.theta,self.ctact[0,1],self.ctact[1,1],self.z[:6])], ['b', 't', 'c', 'z'], ['fb'])
        # pusher
        __fvp = __vp-cs.mtimes(__JA0,__VA)
        self.fvp_ = cs.Function('fvp_', [__Beta,__theta,__yA0,__yA1,__f_ctact,__vp], [cs.vertcat(0,__fvp[1])], ['b', 't', 'ya0', 'ya1', 'f', 'v'], ['fp_'])
        self.fvp = cs.Function('fvp', [self.beta,self.theta,self.ctact,self.z,self.vp],
                                [self.fvp_(self.beta,self.theta,self.ctact[0,1],self.ctact[1,1],self.z[:6],self.vp)], ['b', 't', 'c', 'z', 'v'], ['fp'])
        #  -------------------------------------------------------------------

        # other matrices for debug
        #  -------------------------------------------------------------------
        __F = cs.mtimes(__JA0.T, __f_norm[0]*__nA0.T + cs.mtimes(__DA0, __f_tan[0:2])) + \
              cs.mtimes(__JA1.T, __f_norm[1]*__nA1.T + cs.mtimes(__DA1, __f_tan[2:4]))
        self.F_ = cs.Function('F_', [__yA0,__yA1,__f_ctact], [__F], ['ya0', 'ya1','f'], ['F_'])
        self.F = cs.Function('F', [self.ctact,self.z], [self.F_(self.ctact[0,1],self.ctact[1,1],self.z[:6])], ['c', 'f'], ['F'])
        #  -------------------------------------------------------------------
        self.V_ = cs.Function('V_', [__Beta,__yA0,__yA1,__f_ctact], [__VA], ['b', 'ya0', 'ya1', 'f'], ['V_'])
        self.V = cs.Function('V', [self.beta,self.ctact,self.z], [self.V_(self.beta,self.ctact[0,1],self.ctact[1,1],self.z[:6])], ['b', 'c', 'f'], ['V'])
        #  -------------------------------------------------------------------
        __vpB = cs.mtimes(__K, cs.mtimes(__nsel, __f_norm) + cs.mtimes(__tsel, __f_tan))-cs.mtimes(__JA1,__VA)
        self.vpB_ = cs.Function('vpB_', [__Beta,__theta,__yA0,__yA1,__f_ctact], [cs.vertcat(0, __vpB[1])], ['b', 't', 'ya0', 'ya1', 'f'], ['v'])
        self.vpB = cs.Function('vpB', [self.beta,self.theta,self.ctact,self.z], 
                               [self.vpB_(self.beta,self.theta,self.ctact[0,1],self.ctact[1,1],self.z[:6])], ['b', 't', 'c', 'f'], ['v'])
        #  -------------------------------------------------------------------
        self.N_ = cs.Function('N_', [__yA0,__yA1], [__N], ['ya0', 'ya1'], ['N_'])
        self.N = cs.Function('N', [self.ctact], [self.N_(self.ctact[0,1],self.ctact[1,1])], ['c'], ['N'])
        self.L_ = cs.Function('L_', [__yA0,__yA1], [__L], ['ya0', 'ya1'], ['L_'])
        self.L = cs.Function('L', [self.ctact], [self.L_(self.ctact[0,1],self.ctact[1,1])], ['c'], ['L'])
        #  -------------------------------------------------------------------
        
        # define symbols for building optimization problems
        #  -------------------------------------------------------------------
        if self.controlRelPose:
            self.Nx = 3
            self.x_opt = cs.SX.sym('x', self.Nx)
            self.lbx = [-0.5*self.y_length, -0.5*self.y_length, -cs.inf]
            self.ubx = [0.5*self.y_length, 0.5*self.y_length, cs.inf]
            
            __dx = cs.SX(3,1)
            __dx[0] = cs.mtimes(np.expand_dims(__DA0[:,0], axis=0), __vp - cs.mtimes(__JA0, __VA))
            __dx[1] = cs.mtimes(np.expand_dims(__DA1[:,1], axis=0), cs.mtimes(__TB2A, cs.mtimes(__JB0, __VB))-cs.mtimes(__JA1, __VA))
            __dx[2] = (__VB - __VA)[2]
            
        else:
            self.Nx = 6
            self.x_opt = cs.SX.sym('x', self.Nx)
            self.lbx = [-0.5*self.y_length, -0.5*self.y_length, -cs.inf, -cs.inf, -cs.inf, -cs.inf]
            self.ubx = [0.5*self.y_length, 0.5*self.y_length, 0., cs.inf, cs.inf, cs.inf]
             
            __dx = cs.SX(6,1)
            __dx[0] = cs.mtimes(np.expand_dims(__DA0[:,0], axis=0), __vp - cs.mtimes(__JA0, __VA))
            __dx[1] = cs.mtimes(np.expand_dims(__DA1[:,1], axis=0), cs.mtimes(__TB2A, cs.mtimes(__JB0, __VB))-cs.mtimes(__JA1, __VA))
            __dx[2] = (__VB - __VA)[2]
            __dx[3:] = cs.mtimes(__RA, __VA)
        
        self.Nu = 2
        self.u_opt = cs.SX.sym('u', self.Nu)
        self.lbu = [0., -self.v_lim]
        self.ubu = [self.v_lim, self.v_lim]
        
        # complementary variables
        self.Nz = 8
        self.z_opt = cs.SX.sym('z', self.Nz)
        self.lbz = [0.]*self.Nz
        self.ubz = [self.f_lim]*(self.Nz-2)+[cs.inf]*2
        self.Nw = 8
        self.w_opt = cs.SX.sym('w', self.Nw)
        self.lbw = [0.]*self.Nw
        self.ubw = [cs.inf]*self.Nw
        
        # slack variables
        self.Ns = 8
        self.s_opt = cs.SX.sym('s', self.Ns)
        self.s0 = [1.]*self.Ns
        self.lbs = [-cs.inf]*self.Ns
        self.ubs = [cs.inf]*self.Ns

        self.M_opt = cs.Function('M_opt', [self.beta, self.x_opt], [self.M_opt_(self.beta, cs.vertcat(0., self.x_opt[2]), self.x_opt[0], self.x_opt[1])], ['b', 'x'], ['M_opt'])
        self.q_opt = cs.Function('q_opt', [self.u_opt], [self.q_opt_(self.u_opt)], ['u'], ['q_opt'])
        
        self.f_opt_ = cs.Function('f_opt_', [__Beta, __theta, __yA0, __yA1, __vp, __f_ctact], [__dx], ['b', 't', 'ya0', 'ya1', 'vp', 'f'], ['f_opt_'])
        self.f_opt = cs.Function('f_opt', [self.beta, self.x_opt, self.u_opt, self.z_opt],
                                 [self.f_opt_(self.beta, cs.vertcat(0., self.x_opt[2]), self.x_opt[0], self.x_opt[1], self.u_opt, self.z_opt[:6])],
                                 ['b', 'x', 'u', 'z'], ['f_opt'])
        
        # (complementary) constraints
        self.Ng_u = 8
        g_zw_ = self.z_opt[0]*self.w_opt[0] + self.s_opt[0]
        for i in range(1, self.Ng_u):
            g_zw_ = cs.vertcat(g_zw_, self.z_opt[i]*self.w_opt[i] + self.s_opt[i])
        self.g_zw = cs.Function('g_u', [self.z_opt, self.w_opt, self.s_opt], [g_zw_], ['z', 'w', 's'], ['g'])
        self.g_lb = [0.]*self.Ng_u
        self.g_ub = [0.]*self.Ng_u
        
        # cost gain for state variable
        __i_th = cs.SX.sym('__i_th')
        __Kx_max = self.Kx_max
        self.kx_f = cs.Function('kx', [__i_th], [__Kx_max * (1. / __i_th)])
        
        # cost gain for slack variable
        __Ks_max = self.Ks_max
        __Ks_min = self.Ks_min
        self.ks_f = cs.Function('ks', [__i_th], [__Ks_max * cs.exp(__i_th * cs.log(__Ks_min / __Ks_max))])
        #  -------------------------------------------------------------------
        
    def set_patches(self, ax, x_a_data, x_b_data, x_c_data, beta):
        """
        :param vis_flatness: visualize auxiliary lines for differential flatness or not
        """
        Xl = beta[0]
        Yl = beta[1]
        R_pusher = beta[2]
        x_a0 = x_a_data[:, 0]
        x_b0 = x_b_data[:, 0]
        
        R_a0 = np.eye(3)
        d_a0 = R_a0.dot(np.array([-Xl/2., -Yl/2., 0]))        
        R_b0 = np.eye(3)
        d_b0 = R_b0.dot(np.array([-Xl/2., -Yl/2., 0]))
        
        x_c0 = x_a0[:2] + R_a0[:2, :2] @ x_c_data[:, 0]
        d_c0 = R_a0.dot(np.array([-R_pusher, 0., 0.]))
        
        self.slider_a = patches.Rectangle(
                x_a0[0:2]+d_a0[0:2], Xl, Yl, angle=0.0)
        self.slider_b = patches.Rectangle(
                x_b0[0:2]+d_b0[0:2], Xl, Yl, angle=0.0)
        
        self.pusher = patches.Circle(
                x_c0[0:2]+d_c0[0:2], radius=R_pusher, color='black')
        
        # self.path_past, = ax.plot(x0[0], x0[1], color='orange')
        # self.path_future, = ax.plot(x0[0], x0[1],
        #         color='orange', linestyle='dashed')
        # self.cor = patches.Circle(
        #         np.array([0, 0]), radius=0.002, color='deepskyblue')
        
        ax.add_patch(self.slider_a)
        ax.add_patch(self.slider_b)
        ax.add_patch(self.pusher)
        
        # ax.add_patch(self.cor)
        # self.path_past.set_linewidth(2)
        
    def animate(self, i, ax, x_a_data, x_b_data, x_c_data, beta, X_future=None):
        Xl = beta[0]
        Yl = beta[1]
        R_pusher = beta[2]
        x_ai = x_a_data[:, i]
        x_bi = x_b_data[:, i]
        # distance between centre of square reference corner
        R_ai = np.array(self.RA(x_ai[2]))
        R_bi = np.array(self.RB(x_bi[2]))
        d_ai = R_ai.dot(np.array([-Xl/2, -Yl/2, 0]))
        d_bi = R_bi.dot(np.array([-Xl/2, -Yl/2, 0]))
        x_ci = x_ai[:2] + R_ai[:2, :2] @ x_c_data[:, i]
        d_ci = R_ai.dot(np.array([-R_pusher, 0., 0.]))
        # square reference corner
        c_ai = x_ai[0:3] + d_ai
        c_bi = x_bi[0:3] + d_bi
        c_ci = x_ci[:2] + d_ci[:2]
        # compute transformation with respect to rotation angle xi[2]
        trans_a_ax = ax.transData
        coords_a = trans_a_ax.transform(c_ai[0:2])
        trans_ai = transforms.Affine2D().rotate_around(
                coords_a[0], coords_a[1], x_ai[2])
        trans_b_ax = ax.transData
        coords_b = trans_b_ax.transform(c_bi[0:2])
        trans_bi = transforms.Affine2D().rotate_around(
                coords_b[0], coords_b[1], x_bi[2])
        # Set changes
        self.slider_a.set_transform(trans_a_ax+trans_ai)
        self.slider_a.set_xy([c_ai[0], c_ai[1]])
        self.slider_b.set_transform(trans_b_ax+trans_bi)
        self.slider_b.set_xy([c_bi[0], c_bi[1]])
        self.pusher.set_center(c_ci)
        
        # Set path changes
        # if self.path_past is not None:
        #     self.path_past.set_data(x_data[0, 0:i], x_data[1, 0:i])
        # if (self.path_future is not None) and (X_future is not None):
        #     self.path_future.set_data(X_future[0, :, i], X_future[1, :, i])
            
        return []

# -------------------------------------------------------------------

# if __name__ == '__main__':
#     planning_config = sliding_pack.load_config('planning_config.yaml')
#     dyn = Double_Sys_sq_slider_quasi_static_ellip_lim_surf(planning_config['dynamics'], contactNum=2)
#     import pdb; pdb.set_trace()

# -------------------------------------------------------------------
