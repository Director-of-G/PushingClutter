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
    # The slider A has a face contact. and the slider B has a vertex contact.
    # The dynamic model is under quasi-static assumption.
    # The dynamic model is approximated by an ellipsoid.
    def __init__(self, configDict, contactNum=2.):

        # init parameters
        #  -------------------------------------------------------------------
        self.miu = configDict['pusherFricCoef']  # fric between pusher and slider
        self.f_lim = configDict['pusherForceLim']
        self.n_ctact = contactNum  # number of contacts (including the pusher)
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
        # R - rotation matrices
        __RA, __RB = cs.SX(3, 3), cs.SX(3, 3)
        __cthetaA = cs.cos(__thetaA); __sthetaA = cs.sin(__thetaA)
        __RA[0,0] = __cthetaA; __RA[0,1] = -__sthetaA; __RA[1,0] = __sthetaA; __RA[1,1] = __cthetaA; __RA[2,2] = 1.0
        __cthetaB = cs.cos(__thetaB); __sthetaB = cs.sin(__thetaB)
        __RB[0,0] = __cthetaB; __RB[0,1] = -__sthetaB; __RB[1,0] = __sthetaB; __RB[1,1] = __cthetaB; __RB[2,2] = 1.0
        # T - planar transformation matrices
        __TB2A = cs.mtimes(cs.inv(__RA[:2,:2]), __RB[:2,:2]); __TA2B = cs.mtimes(cs.inv(__RB[:2,:2]), __RA[:2,:2])
        # -------------------------------------------------------------------

        # contact jacobian
        #  -------------------------------------------------------------------
        # contact locations
        __xA0 = cs.SX.sym('__xA0'); __yA0 = cs.SX.sym('__yA0')
        __xA1 = cs.SX.sym('__xA1'); __yA1 = cs.SX.sym('__yA1')
        __xB0 = cs.SX.sym('__xB0'); __yB0 = cs.SX.sym('__yB0')
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
        self.K = cs.Function('K', [__Beta, __ctact, __theta], [__K], ['b', 'c', 't'], ['K'])
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
        self.M_ = cs.Function('M_', [__Beta, __theta, __ctact],
                              [__M], ['b', 't', 'c'], ['M_'])
        self.M = cs.Function('M', [self.beta, self.theta, self.ctact],
                             [self.M_(self.beta, self.theta, self.ctact)], ['b', 't', 'c'], ['M'])
        self.q_ = cs.Function('q_', [__vp], [__q], ['v'], ['q'])
        self.q = cs.Function('q', [self.vp], [self.q_(self.vp)], ['v'], ['q'])
        #  -------------------------------------------------------------------

        # dynamic functions
        #  -------------------------------------------------------------------
        # sliderA twist
        __V = cs.mtimes(__A, cs.mtimes(__N.T, __f_norm) + cs.mtimes(__L.T, __f_tan))
        # slider pose
        __fA = cs.mtimes(__RA,__V)
        __fB = cs.mtimes(cs.mtimes(__RB,__A),cs.mtimes(__JB0.T,cs.mtimes(__TA2B,cs.mtimes(__nsel,__f_norm)+cs.mtimes(__tsel,__f_tan))))
        self.fA_ = cs.Function('fA_', [__Beta, __theta, __ctact, __f_ctact], [__fA], ['b', 't', 'f', 'c'], ['fa_'])
        self.fB_ = cs.Function('fB_', [__Beta, __theta, __ctact, __f_ctact], [__fB], ['b', 't', 'f', 'c'], ['fb_'])
        self.fA = cs.Function('fA', [self.beta, self.theta, self.ctact, self.z],
                            [self.fA_(self.beta, self.theta, self.ctact, self.z[:6])], ['b', 't', 'f', 'c'], ['fa'])
        self.fB = cs.Function('fB', [self.beta, self.theta, self.ctact, self.z],
                            [self.fA_(self.beta, self.theta, self.ctact, self.z[:6])], ['b', 't', 'f', 'c'], ['fb'])
        # pusher
        __fvp = cs.mtimes(__JA0,__V)-__vp
        self.fvp_ = cs.Function('fvp_', [__Beta, __theta, __ctact, __f_ctact], [cs.vertcat(0,__fvp[1])], ['b', 't', 'c', 'f'], ['fp'])
        self.fvp_ = cs.Function('fvp_', [self.beta, self.theta, self.ctact, self.z],
                                [self.fvp_(self.beta,self.theta,self.ctact,self.z[:6])], ['b', 't', 'c', 'f'], ['fp'])
        #  -------------------------------------------------------------------

        # other matrices for debug
        #  -------------------------------------------------------------------
        __F = cs.mtimes(__JA0.T, __f_norm[0]*__nA0.T + cs.mtimes(__DA0, __f_tan[0:2])) + \
              cs.mtimes(__JA1.T, __f_norm[1]*__nA1.T + cs.mtimes(__DA1, __f_tan[2:4]))
        self.F_ = cs.Function('F_', [__ctact, __f_ctact], [__F], ['c', 'f'], ['F_'])
        self.F = cs.Function('F', [self.ctact, self.z], [self.F_(self.ctact, self.z[:6])], ['c', 'f'], ['F'])
        #  -------------------------------------------------------------------
        self.V_ = cs.Function('V_', [__Beta, __ctact, __f_ctact], [__V], ['b', 'c', 'f'], ['V_'])
        self.V = cs.Function('V', [self.beta, self.ctact, self.z], [self.V_(self.beta, self.ctact, self.z[:6])], ['b', 'c', 'f'], ['V'])
        #  -------------------------------------------------------------------
        __vpB = cs.mtimes(__K, cs.mtimes(__nsel, __f_norm) + cs.mtimes(__tsel, __f_tan))
        self.vpB_ = cs.Function('vpB_', [__Beta, __theta, __ctact, __f_ctact], [__vpB], ['b', 't', 'c', 'f'], ['v'])
        self.vpB = cs.Function('vpB', [self.beta, self.theta, self.ctact, self.z], 
                               [self.vpB_(self.beta, self.theta, self.ctact, self.z[:6])], ['b', 't', 'c', 'f'], ['v'])
        #  -------------------------------------------------------------------
        self.N_ = cs.Function('N_', [__ctact], [__N], ['c'], ['N_'])
        self.N = cs.Function('N', [self.ctact], [self.N_(self.ctact)], ['c'], ['N'])
        self.L_ = cs.Function('L_', [__ctact], [__L], ['c'], ['L_'])
        self.L = cs.Function('L', [self.ctact], [self.L_(self.ctact)], ['c'], ['L'])
        #  -------------------------------------------------------------------

# -------------------------------------------------------------------

# if __name__ == '__main__':
#     planning_config = sliding_pack.load_config('planning_config.yaml')
#     dyn = Double_Sys_sq_slider_quasi_static_ellip_lim_surf(planning_config['dynamics'], contactNum=2)
#     import pdb; pdb.set_trace()

# -------------------------------------------------------------------
