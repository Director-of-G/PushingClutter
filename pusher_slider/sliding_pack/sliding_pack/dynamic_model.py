# Author: Joao Moura (Modifed by Yongpeng Jiang)
# Contact: jpousad@ed.ac.uk (jyp19@mails.tsinghua.edu.cn)
# Date: 19/10/2020 (Modified on 15/12/2022)
# -------------------------------------------------------------------
# Description:
# 
# Functions modelling the dynamics of an object sliding on a table.
# Based on: Hogan F.R, Rodriguez A. (2020) IJRR paper
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------
import sys
import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import casadi as cs
import sliding_pack
# -------------------------------------------------------------------

class Sys_sq_slider_quasi_static_ellip_lim_surf():
    # The dynamic model for single-pusher-single-slider.
    # The dynamic model is under quasi-static assumption.
    # The dynamic model is approximated by an ellipsoid.
    def __init__(self, configDict, contactMode='sticking', contactFace='-x', pusherAngleLim=0.):

        # init parameters
        self.mode = contactMode
        self.face = contactFace
        # self.sl = configDict['sideLenght']  # side dimension of the square slider [m]
        self.miu = configDict['pusherFricCoef']  # fric between pusher and slider
        self.f_lim = configDict['pusherForceLim']
        self.psi_dot_lim = configDict['pusherAngleVelLim']
        self.Kz_max = configDict['Kz_max']
        self.Kz_min = configDict['Kz_min']
        #  -------------------------------------------------------------------
        # vector of physical parameters
        # self.beta = [self.xl, self.yl, self.r_pusher]
        
        # obstacles
        self.Radius = 0.05
        
        self.Nbeta = 3
        self.beta = cs.SX.sym('beta', self.Nbeta)
        # beta[0] - xl
        # beta[1] - yl
        # beta[2] - r_pusher
        #  -------------------------------------------------------------------
        # self.psi_lim = 0.9*cs.arctan2(self.beta[0], self.beta[1])
        if self.mode == 'sticking':
            self.psi_lim = pusherAngleLim
        else:
            if self.face == '-x' or self.face == '+x':
                self.psi_lim = configDict['xFacePsiLimit']
            elif self.face == '-y' or self.face == '+y':
                self.psi_lim = configDict['yFacePsiLimit']
                # self.psi_lim = 0.405088
                # self.psi_lim = 0.52

        # system constant variables
        self.Nx = 4  # number of state variables

        # vectors of state and control
        #  -------------------------------------------------------------------
        # x - state vector
        # x[0] - x slider CoM position in the global frame
        # x[1] - y slider CoM position in the global frame
        # x[2] - slider orientation in the global frame
        # x[3] - angle of pusher relative to slider
        self.x = cs.SX.sym('x', self.Nx)
        # dx - derivative of the state vector
        self.dx = cs.SX.sym('dx', self.Nx)
        #  -------------------------------------------------------------------

        # auxiliar symbolic variables
        # used to compute the symbolic representation for variables
        # -------------------------------------------------------------------
        # x - state vector
        __x_slider = cs.SX.sym('__x_slider')  # in global frame [m]
        __y_slider = cs.SX.sym('__y_slider')  # in global frame [m]
        __theta = cs.SX.sym('__theta')  # in global frame [rad]
        __psi = cs.SX.sym('__psi')  # in relative frame [rad]
        __x = cs.veccat(__x_slider, __y_slider, __theta, __psi)
        # u - control vector
        __f_norm = cs.SX.sym('__f_norm')  # in local frame [N]
        __f_tan = cs.SX.sym('__f_tan')  # in local frame [N]
        # rel vel between pusher and slider [rad/s]
        __psi_dot = cs.SX.sym('__psi_dot')
        __u = cs.veccat(__f_norm, __f_tan, __psi_dot)
        # beta - dynamic parameters
        __xl = cs.SX.sym('__xl')  # slider x lenght
        __yl = cs.SX.sym('__yl')  # slider y lenght
        __r_pusher = cs.SX.sym('__r_pusher')  # radious of the cilindrical pusher
        __beta = cs.veccat(__xl, __yl, __r_pusher)

        # system model
        # -------------------------------------------------------------------
        # Rotation matrix
        __Area = __xl*__yl
        __int_Area = sliding_pack.integral.rect_cs(__xl, __yl)
        __c = __int_Area/__Area # ellipsoid approximation ratio
        self.c = cs.Function('c', [__x, __beta], [__c], ['x', 'b'], ['c'])
        __A = cs.SX.sym('__A', cs.Sparsity.diag(3))
        __A[0,0] = __A[1,1] = 1.; __A[2,2] = 1./(__c**2)
        self.A = cs.Function('A', [__beta], [__A], ['b'], ['A'])
        __ctheta = cs.cos(__theta)
        __stheta = cs.sin(__theta)
        __R = cs.SX(3, 3)  # anti-clockwise rotation matrix (from {Slider} to {World})
        __R[0,0] = __ctheta; __R[0,1] = -__stheta; __R[1,0] = __stheta; __R[1,1] = __ctheta; __R[2,2] = 1.0;
        #  -------------------------------------------------------------------
        self.R = cs.Function('R', [__x], [__R], ['x'], ['R'])  # (rotation matrix from {Slider} to {World})
        #  -------------------------------------------------------------------
        __p = cs.SX.sym('p', 2) # pusher position
        __rc_prov = cs.mtimes(__R[0:2,0:2].T, __p - __x[0:2])  # (Real {Pusher Center} in {Slider})
        #  -------------------------------------------------------------------
        # slider frame ({x} forward, {y} left)
        # slider position
        # if self.face == '-x':
        __xc = -__xl/2; __yc = -(__xl/2)*cs.tan(__psi)  # ({Contact Point} in {Slider})
        __rc = cs.SX(2,1); __rc[0] = __xc-__r_pusher; __rc[1] = __yc  # ({Pusher Center} in {Slider})
        __ctact = cs.SX(2,1); __ctact[0] = __xc; __ctact[1] = __yc  # ({Contact Point} in {Slider})
        #  -------------------------------------------------------------------
        __psi_prov = -cs.atan2(__rc_prov[1], __xl/2)  # (Real {φ_c})
        # elif self.face == '+x':
        #     __xc = __xl/2; __yc = __xl/2*cs.tan(__psi)  # ({Contact Point} in {Slider})
        #     __rc = cs.SX(2,1); __rc[0] = __xc+__r_pusher; __rc[1] = __yc  # ({Pusher Center} in {Slider})
        #     #  -------------------------------------------------------------------
        #     __psi_prov = -cs.atan2(__rc_prov[1], -__xl/2)  # (Real {φ_c})
        # elif self.face == '-y' or self.face == '+y':
        #     __xc = -(__yl/2)/cs.tan(__psi) if np.abs(__psi - 0.5 * np.pi) > 1e-3 else 0.; __yc = -__yl/2  # ({Contact Point} in {Slider})
        #     __rc = cs.SX(2,1); __rc[0] = __xc; __rc[1] = __yc-__r_pusher  # ({Pusher Center} in {Slider})
        #     #  -------------------------------------------------------------------
        #     __psi_prov = -cs.atan2(__yl/2, __rc_prov[0]) + cs.pi  # (Real {φ_c})
        # else:
        #     __xc = (__yl/2)/cs.tan(__psi) if np.abs(__psi + 0.5 * np.pi) > 1e-3 else 0.; __yc = __yl/2  # ({Contact Point} in {Slider})
        #     __rc = cs.SX(2,1); __rc[0] = __xc; __rc[1] = __yc+__r_pusher  # ({Pusher Center} in {Slider})
        #     #  -------------------------------------------------------------------
        #     __psi_prov = -cs.atan2(-__yl/2, __rc_prov[0]) - cs.pi  # (Real {φ_c})
            
        # pusher position
        __p_pusher = cs.mtimes(__R[0:2,0:2], __rc)[0:2] + __x[0:2]  # ({Pusher Center} in {World})
        __p_ctact = cs.mtimes(__R[0:2,0:2], __ctact)[0:2] + __x[0:2]  # ({Contact Point} in {World})
        #  -------------------------------------------------------------------
        self.psi_ = cs.Function('psi_', [__x,__p,__beta], [__psi_prov])  # compute (φ_c) from state variables, pusher coordinates and slider geometry
        self.psi = cs.Function('psi', [self.x,__p,self.beta], [self.psi_(self.x, __p, self.beta)])
        #  -------------------------------------------------------------------
        self.p_ = cs.Function('p_', [__x,__beta], [__p_pusher], ['x', 'b'], ['p'])  # compute (pusher_center_coordinate) from state variables and slider geometry
        self.p = cs.Function('p', [self.x, self.beta], [self.p_(self.x, self.beta)], ['x', 'b'], ['p'])
        #  -------------------------------------------------------------------
        self.ctact_ = cs.Function('ctact_', [__x,__beta], [__ctact], ['x', 'b'], ['ctact'])
        self.ctact = cs.Function('ctact', [__x,__beta], [self.ctact_(self.x, self.beta)], ['x', 'b'], ['ctact'])
        self.p_ctact_ = cs.Function('p_ctact_', [__x,__beta], [__p_ctact], ['x', 'b'], ['p_ctact'])  # compute (pusher_center_coordinate) from state variables and slider geometry
        self.p_ctact = cs.Function('p_ctact', [self.x, self.beta], [self.p_ctact_(self.x, self.beta)], ['x', 'b'], ['p_ctact'])
        #  -------------------------------------------------------------------
        self.s = cs.Function('s', [self.x], [self.x[0:3]], ['x'], ['s'])  # compute (x, y, θ) from state variables
        #  -------------------------------------------------------------------
        
        # dynamics
        __Jc = cs.SX(2,3)
        # if self.face == '-x':
        __Jc[0,0] = 1; __Jc[1,1] = 1; __Jc[0,2] = -__yc; __Jc[1,2] = __xc;  # contact jacobian
        # elif self.face == '+x':
        #     __Jc[0,0] = -1; __Jc[1,1] = -1; __Jc[0,2] = __yc; __Jc[1,2] = -__xc;  # contact jacobian
        # elif self.face == '-y':
        #     __Jc[0,1] = -1; __Jc[1,0] = 1; __Jc[0,2] = __xc; __Jc[1,2] = __yc;  # contact jacobian
        # else:
        #     __Jc[0,1] = 1; __Jc[1,0] = -1; __Jc[0,2] = -__xc; __Jc[1,2] = -__yc;  # contact jacobian
        
        self.RAJc = cs.Function('RAJc', [__x,__beta], [cs.mtimes(cs.mtimes(__R, __A), __Jc.T)], ['x', 'b'], ['f'])
        __f = cs.SX(cs.vertcat(cs.mtimes(cs.mtimes(__R,__A),cs.mtimes(__Jc.T,__u[0:2])),__u[2]))
        #  -------------------------------------------------------------------
        self.f_ = cs.Function('f_', [__x,__u,__beta], [__f], ['x', 'u', 'b'], ['f'])  # compute (f(x, u)) from state variables, input variables and slider geometry
        #  -------------------------------------------------------------------

        # control constraints
        #  -------------------------------------------------------------------
        if self.mode == 'sliding_cc':
            # complementary constraint
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            # u[2] - rel sliding vel between pusher and slider counterclockwise(φ_c(-))
            # u[3] - rel sliding vel between pusher and slider clockwise(φ_c(+))
            self.Nu = 4  # number of action variables
            self.u = cs.SX.sym('u', self.Nu)
            self.Nz = 0
            self.z0 = []
            self.lbz = []
            self.ubz = []
            # discrete extra variable
            self.z_discrete = False
            empty_var = cs.SX.sym('empty_var')
            self.g_u = cs.Function('g_u', [self.u, empty_var], [cs.vertcat(
                # friction cone edges
                self.miu*self.u[0]+self.u[1],  # lambda(+)>=0
                self.miu*self.u[0]-self.u[1],  # lambda(-)>=0
                # complementarity constraint
                (self.miu * self.u[0] - self.u[1])*self.u[3],  # lambda(-)*φ_c(+)=0
                (self.miu * self.u[0] + self.u[1])*self.u[2]  # lambda(+)*φ_c(-)=0
            )], ['u', 'other'], ['g'])
            self.g_lb = [0., 0., 0., 0.]
            self.g_ub = [cs.inf, cs.inf, 0., 0.]
            self.Ng_u = 4
            # cost gain for extra variable
            __Ks_max = self.Kz_max
            __Ks_min = self.Kz_min
            __i_th = cs.SX.sym('__i_th')
            self.kz_f = cs.Function('ks', [__i_th], [__Ks_max * cs.exp(__i_th * cs.log(__Ks_min / __Ks_max))])  # decrease from Ks_max to Ks_min
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
            self.lbu = [0.0,  -self.f_lim, 0.0, 0.0]
            self.ubu = [self.f_lim, self.f_lim, self.psi_dot_lim, self.psi_dot_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u, self.beta], [self.f_(self.x, cs.vertcat(self.u[0:2], self.u[2]-self.u[3]), self.beta)],  ['x', 'u', 'b'], ['f'])
        elif self.mode == 'sliding_cc_slack':
            # complementary constraint + slack variables
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            # u[2] - rel sliding vel between pusher and slider counterclockwise
            # u[3] - rel sliding vel between pusher and slider clockwise
            self.Nu = 4  # number of action variables
            self.u = cs.SX.sym('u', self.Nu)
            self.Nz = 2
            self.z = cs.SX.sym('z', self.Nz)
            self.z0 = [1.]*self.Nz
            self.lbz = [-cs.inf]*self.Nz
            self.ubz = [cs.inf]*self.Nz
            # discrete extra variable
            self.z_discrete = False
            self.g_u = cs.Function('g_u', [self.u, self.z], [cs.vertcat(
                # friction cone edges
                self.miu*self.u[0]+self.u[1],
                self.miu*self.u[0]-self.u[1],
                # complementarity constraint
                (self.miu * self.u[0] - self.u[1])*self.u[3] + self.z[0],
                (self.miu * self.u[0] + self.u[1])*self.u[2] + self.z[1]
            )], ['u', 'other'], ['g'])
            self.g_lb = [0., 0., 0., 0.]
            self.g_ub = [cs.inf, cs.inf, 0., 0.]
            self.Ng_u = 4
            # cost gain for extra variable
            __Ks_max = self.Kz_max
            __Ks_min = self.Kz_min
            __i_th = cs.SX.sym('__i_th')
            self.kz_f = cs.Function('ks', [__i_th], [__Ks_max * cs.exp(__i_th * cs.log(__Ks_min / __Ks_max))])
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
            self.lbu = [0.0,  -self.f_lim, 0.0, 0.0]
            self.ubu = [self.f_lim, self.f_lim, self.psi_dot_lim, self.psi_dot_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u, self.beta], [self.f_(self.x, cs.vertcat(self.u[0:2], self.u[2]-self.u[3]), self.beta)],  ['x', 'u', 'b'], ['f'])
        elif self.mode == 'sliding_mi':
            # mixed integer
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            # u[2] - rel sliding vel between pusher and slider
            self.Nu = 3  # number of action variables
            self.u = cs.SX.sym('u', self.Nu)
            self.Nz = 3
            self.z = cs.SX.sym('z', self.Nz)
            self.z0 = [0]*self.Nz
            self.lbz = [0]*self.Nz
            self.ubz = [1]*self.Nz
            # discrete extra variable
            self.z_discrete = True
            self.Ng_u = 7
            bigM = 500  # big M for the Mixed Integer optimization
            self.g_u = cs.Function('g_u', [self.u, self.z], [cs.vertcat(
                self.miu*self.u[0]+self.u[1] + bigM*self.z[1],  # friction cone edge
                self.miu*self.u[0]-self.u[1] + bigM*self.z[2],  # friction cone edge
                self.miu*self.u[0]+self.u[1] - bigM*(1-self.z[2]),  # friction cone edge
                self.miu*self.u[0]-self.u[1] - bigM*(1-self.z[1]),  # friction cone edge
                self.u[2] + bigM*self.z[2],  # relative rot constraint
                self.u[2] - bigM*self.z[1],
                cs.sum1(self.z),  # sum of the integer variables
            )], ['u', 'other'], ['g'])
            self.g_lb = [0., 0., -cs.inf, -cs.inf, 0., -cs.inf, 1.]
            self.g_ub = [cs.inf, cs.inf, 0., 0., cs.inf, 0., 1.]
            __i_th = cs.SX.sym('__i_th')
            self.kz_f = cs.Function('ks', [__i_th], [0.])
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
            self.lbu = [0.0,  -self.f_lim, 0.0]
            self.ubu = [self.f_lim, self.f_lim, self.psi_dot_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u, self.beta], [self.f_(self.x, self.u, self.beta)],  ['x', 'u', 'b'], ['f'])
        elif self.mode == 'sticking':
            # sticking constraint
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            self.Nu = 2  # number of action variables
            self.u = cs.SX.sym('u', self.Nu)
            empty_var = cs.SX.sym('empty_var')
            self.g_u = cs.Function('g_u', [self.u, empty_var], [cs.vertcat(
                self.miu*self.u[0]+self.u[1],  # friction cone edge
                self.miu*self.u[0]-self.u[1]  # friction cone edge
            )], ['u', 'other'], ['g'])
            self.g_lb = [0.0, 0.0]
            self.g_ub = [cs.inf, cs.inf]
            self.Nz = 0
            self.z0 = []
            self.lbz = []
            self.ubz = []
            # discrete extra variable
            self.z_discrete = False
            self.Ng_u = 2
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, self.psi_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
            self.lbu = [0.0,  -self.f_lim]
            self.ubu = [self.f_lim, self.f_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u, self.beta], [self.f_(self.x, cs.vertcat(self.u, 0.0), self.beta)],  ['x', 'u', 'b'], ['f'])
        else:
            print('Specified mode ``{}`` does not exist!'.format(self.mode))
            sys.exit(-1)
        #  -------------------------------------------------------------------

    def set_patches(self, ax, x_data, beta, vis_flatness=False, u_data=None):
        """
        :param vis_flatness: visualize auxiliary lines for differential flatness or not
        """
        Xl = beta[0]
        Yl = beta[1]
        R_pusher = beta[2]
        x0 = x_data[:, 0]
        # R0 = np.array(self.R(x0))
        R0 = np.eye(3)
        d0 = R0.dot(np.array([-Xl/2., -Yl/2., 0]))
        self.slider = patches.Rectangle(
                x0[0:2]+d0[0:2], Xl, Yl, angle=0.0)
        self.pusher = patches.Circle(
                np.array(self.p(x0, beta)), radius=R_pusher, color='black')
        self.path_past, = ax.plot(x0[0], x0[1], color='orange')
        self.path_future, = ax.plot(x0[0], x0[1],
                color='orange', linestyle='dashed')
        self.cor = patches.Circle(
                np.array([0, 0]), radius=0.002, color='deepskyblue')
        ax.add_patch(self.slider)
        ax.add_patch(self.pusher)
        ax.add_patch(self.cor)
        self.path_past.set_linewidth(2)

        # Set auxiliary lines
        if vis_flatness:
            # set the lines
            self.mid_axis, = ax.plot(0., 0., color='violet', linestyle='dashed')
            self.flat_line, = ax.plot(0., 0., color='violet', linestyle='dashed')
            self.othog_line, = ax.plot(0., 0., color='violet', linestyle='dashed')
            self.force_line, = ax.plot(0., 0., color='violet', linestyle='dashed')
            self.mid_axis.set_linewidth(2.0)
            self.flat_line.set_linewidth(2.0)
            self.othog_line.set_linewidth(2.0)
            self.force_line.set_linewidth(2.0)

            # self.COG = ax.scatter(0, 0, marker='o')

    def animate(self, i, ax, x_data, beta, vis_flatness=False, u_data=None, X_future=None):
        Xl = beta[0]
        Yl = beta[1]
        xi = x_data[:, i]
        # distance between centre of square reference corner
        Ri = np.array(self.R(xi))
        di = Ri.dot(np.array([-Xl/2, -Yl/2, 0]))
        # square reference corner
        ci = xi[0:3] + di
        # compute transformation with respect to rotation angle xi[2]
        trans_ax = ax.transData
        coords = trans_ax.transform(ci[0:2])
        trans_i = transforms.Affine2D().rotate_around(
                coords[0], coords[1], xi[2])
        # Set changes
        self.slider.set_transform(trans_ax+trans_i)
        self.slider.set_xy([ci[0], ci[1]])
        self.pusher.set_center(np.array(self.p(xi, beta)))
        # Set path changes
        if self.path_past is not None:
            self.path_past.set_data(x_data[0, 0:i], x_data[1, 0:i])
        if (self.path_future is not None) and (X_future is not None):
            self.path_future.set_data(X_future[0, :, i], X_future[1, :, i])

        # Set auxiliary lines
        if vis_flatness:
            # Set auxiliary lines
            ab_ratio = (self.A(beta)[0, 0] / self.A(beta)[2, 2]).toarray().squeeze()
            ui = u_data[:, i]
            fn = ui[0]; ft = ui[1]
            ctact_pt = self.ctact_(xi, beta).toarray().squeeze()
            xc = ctact_pt[0]; yc = ctact_pt[1]  # contact point
            r = np.sqrt(xc ** 2 + yc ** 2)
            xt = -ab_ratio * xc / (r ** 2); yt = -ab_ratio * yc / (r ** 2)  # the intersection of central axis and differential flatness line
            # general form of line: Ax + By = C
            # parameter [A, B] of general formalized auxiliary line
            param_AB = np.array([[xc, yc],
                                 [yc, -xc],
                                 [fn, ft],
                                 [ft, -fn]])
            # parameter [C] of general formalized auxiliary line
            param_C = np.array([[xt * xc + yt * yc],
                                [0],
                                [0],
                                [xc * ft - yc * fn]])
            # key points
            Ri_xy = Ri[:2, :2]
            cor_pt = Ri_xy @ np.linalg.inv(param_AB[[0, 2], :]) @ param_C[[0, 2], :].squeeze() + xi[:2] # center of rotation (COR)
            force_pt = Ri_xy @ np.linalg.inv(param_AB[[2, 3], :]) @ param_C[[2, 3], :].squeeze() + xi[:2]  # point on the line of force
            ctact_pt = Ri_xy @ np.array([xc, yc]) + xi[:2]  # contact point
            tilde_pt = Ri_xy @ np.array([xt, yt]) + xi[:2]  # the intersection of central axis and differential flatness line
            # set the lines
            self.mid_axis.set_data([ctact_pt[0], tilde_pt[0]], [ctact_pt[1], tilde_pt[1]])
            self.flat_line.set_data([tilde_pt[0], cor_pt[0]], [tilde_pt[1], cor_pt[1]])
            self.othog_line.set_data([force_pt[0], cor_pt[0]], [force_pt[1], cor_pt[1]])
            self.force_line.set_data([force_pt[0], ctact_pt[0]], [force_pt[1], ctact_pt[1]])
            self.cor.set_center(np.array([cor_pt[0], cor_pt[1]]))
            print(cor_pt)
            # # set the slider's center
            # self.COG.set_xy(xi[0], xi[1])
        return []
    #  -------------------------------------------------------------------

# -------------------------------------------------------------------

class Aug_Sys_sq_slider_quasi_static_ellip_lim_surf():
    # The augmented dynamic model for single-pusher-multiple-slider.
    # The dynamic model is under quasi-static assumption.
    # The dynamic model is approximated by an ellipsoid.
    # The contacts between adjacent sliders are assumed to be sticking.
    def __init__(self, configDict, contactMode='sticking', contactFace='-x', pusherAngleLim=0., sliderRelAngleLim=0., beta=None):

        # init parameters
        self.mode = contactMode
        self.face = contactFace
        # self.sl = configDict['sideLenght']  # side dimension of the square slider [m]
        self.miu = configDict['pusherFricCoef']  # fric between pusher and slider
        self.f_lim = configDict['pusherForceLim']
        self.psi_dot_lim = configDict['pusherAngleVelLim']
        self.Kz_max = configDict['Kz_max']
        self.Kz_min = configDict['Kz_min']
        #  -------------------------------------------------------------------
        # vector of physical parameters
        # self.beta = [self.xl, self.yl, self.r_pusher]
        
        # obstacles
        self.Radius = 0.05
        
        self.Nbeta = 3
        self.beta = cs.SX.sym('beta', self.Nbeta)  # symbolic beta
        # beta[0] - xl
        # beta[1] - yl
        # beta[2] - r_pusher
        #  -------------------------------------------------------------------
        # self.psi_lim = 0.9*cs.arctan2(self.beta[0], self.beta[1])
        if self.mode == 'sticking':
            self.psi_lim = pusherAngleLim
        else:
            if self.face == '-x' or self.face == '+x':
                self.psi_lim = configDict['xFacePsiLimit']
            elif self.face == '-y' or self.face == '+y':
                self.psi_lim = configDict['yFacePsiLimit']
                # self.psi_lim = 0.405088
                # self.psi_lim = 0.52
        self.psi_r_lim = sliderRelAngleLim

        # initialize beta, when numerizing some symbols, it is used
        try:
            assert beta is not None
            self.beta_eval = beta  # numerical beta
        except:
            print('Beta should be assigned a value for further numerization!')
            sys.exit(-1)

        # system constant variables
        self.Nx = 8  # number of state variables

        # vectors of state and control
        #  -------------------------------------------------------------------
        # x - state vector
        # x[0] - x slider1 CoM position in the global frame
        # x[1] - y slider1 CoM position in the global frame
        # x[2] - slider1 orientation in the global frame
        # x[3] - angle of pusher relative to slider1
        # x[4] - x slider2 CoM position in the global frame
        # x[5] - y slider2 CoM position in the global frame
        # x[6] - slider2 orientation in the global frame
        # x[7] - angle of slider2 relative to slider1
        self.x = cs.SX.sym('x', self.Nx)
        # dx - derivative of the state vector
        self.dx = cs.SX.sym('dx', self.Nx)
        #  -------------------------------------------------------------------

        # auxiliar symbolic variables
        # used to compute the symbolic representation for variables
        # -------------------------------------------------------------------
        # x - state vector
        __x_slider1 = cs.SX.sym('__x_slider1')  # in global frame [m]
        __y_slider1 = cs.SX.sym('__y_slider1')  # in global frame [m]
        __theta1 = cs.SX.sym('__theta1')  # in global frame [rad]
        __psi_c = cs.SX.sym('__psi_c')  # in slider1 frame [rad]
        __x_slider2 = cs.SX.sym('__x_slider2')  # in global frame [m]
        __y_slider2 = cs.SX.sym('__y_slider2')  # in global frame [m]
        __theta2 = cs.SX.sym('__theta2')  # in global frame [rad]
        __psi_r = cs.SX.sym('__psi_r')  # in slider1 frame [rad]
        __x = cs.veccat(__x_slider1, __y_slider1, __theta1, __psi_c, __x_slider2, __y_slider2, __theta2, __psi_r)
        # u - control vector
        __f_norm = cs.SX.sym('__f_norm')  # in slider1 frame [N]
        __f_tan = cs.SX.sym('__f_tan')  # in slider1 frame [N]
        # uL - auxiliar control vector, corresponding to the left contact point on slider1
        __fL_norm = cs.SX.sym('__fL_norm')  # in slider1 frame [N]
        __fL_tan = cs.SX.sym('__fL_tan')  # in slider1 frame [N]
        # uR - auxiliar control vector, corresponding to the right contact point on slider1
        __fR_norm = cs.SX.sym('__fR_norm')  # in slider1 frame [N]
        __fR_tan = cs.SX.sym('__fR_tan')  # in slider1 frame [N]
        # rel vel between pusher and slider [rad/s]
        __psi_dot = cs.SX.sym('__psi_dot')
        __u = cs.veccat(__f_norm, __f_tan, __psi_dot, __fL_norm, __fL_tan, __fR_norm, __fR_tan)
        # beta - dynamic parameters
        __xl = cs.SX.sym('__xl')  # slider x lenght
        __yl = cs.SX.sym('__yl')  # slider y lenght
        __r_pusher = cs.SX.sym('__r_pusher')  # radious of the cilindrical pusher
        __beta = cs.veccat(__xl, __yl, __r_pusher)

        # system model
        # -------------------------------------------------------------------
        # Rotation matrix
        __Area = __xl*__yl
        __int_Area = sliding_pack.integral.rect_cs(__xl, __yl)
        __c = __int_Area/__Area # ellipsoid approximation ratio
        self.c = cs.Function('c', [__x, __beta], [__c], ['x', 'b'], ['c'])
        __A = cs.SX.sym('__A', cs.Sparsity.diag(3))
        __A[0,0] = __A[1,1] = 1.; __A[2,2] = 1./(__c**2)
        self.A = cs.Function('A', [__beta], [__A], ['b'], ['A'])
        __ctheta1 = cs.cos(__theta1)
        __stheta1 = cs.sin(__theta1)
        __R = cs.SX(3, 3)  # anti-clockwise rotation matrix (from {Slider1} to {World})
        __R[0,0] = __ctheta1; __R[0,1] = -__stheta1; __R[1,0] = __stheta1; __R[1,1] = __ctheta1; __R[2,2] = 1.0
        #  -------------------------------------------------------------------
        self.R = cs.Function('R', [__x], [__R], ['x'], ['R'])  # (rotation matrix from {Slider1} to {World})
        #  -------------------------------------------------------------------
        __p = cs.SX.sym('p', 2) # pusher position
        __rc_prov = cs.mtimes(__R[0:2,0:2].T, __p - __x[0:2])  # (Real {Pusher Center} in {Slider1})
        #  -------------------------------------------------------------------
        # slider frame ({x} forward, {y} left)
        # pusher position
        __xc = -__xl/2; __yc = -(__xl/2)*cs.tan(__psi_c)  # ({Contact Point} in {Slider1})
        __rc = cs.SX(2,1); __rc[0] = __xc-__r_pusher; __rc[1] = __yc  # ({Pusher Center} in {Slider1})
        __ctact = cs.SX(2,1); __ctact[0] = __xc; __ctact[1] = __yc  # ({Contact Point} in {Slider1})
        # Left contact point position
        __xL = __xl/2; __yL=(__xl/2)*((__yl/__xl)+2*cs.tan(cs.fmin(__psi_r, 0)))  # ({Left Contact Point} in {Slider1})
        __Ltact = cs.SX(2,1); __Ltact[0] = __xL; __Ltact[1] = __yL  # ({Left Contact Point} in {Slider1})
        # Right contact point position
        __xR = __xl/2; __yR=(__xl/2)*(-(__yl/__xl)+2*cs.tan(cs.fmax(__psi_r, 0)))  # ({Right Contact Point} in {Slider1})
        __Rtact = cs.SX(2,1); __Rtact[0] = __xR; __Rtact[1] = __yR  # ({Right Contact Point} in {Slider1})
        #  -------------------------------------------------------------------
        __psi_prov = -cs.atan2(__rc_prov[1], __xl/2)  # (Real {φ_c})
            
        # pusher position
        __p_pusher = cs.mtimes(__R[0:2,0:2], __rc)[0:2] + __x[0:2]  # ({Pusher Center} in {World})
        __p_ctact = cs.mtimes(__R[0:2,0:2], __ctact)[0:2] + __x[0:2]  # ({Contact Point} in {World})
        #  -------------------------------------------------------------------
        self.psi_c_ = cs.Function('psi_c_', [__x,__p,__beta], [__psi_prov])  # compute (φ_c) from state variables, pusher coordinates and slider geometry
        self.psi_c = cs.Function('psi_c', [self.x,__p,self.beta], [self.psi_c_(self.x, __p, self.beta)])
        #  -------------------------------------------------------------------
        self.p_ = cs.Function('p_', [__x,__beta], [__p_pusher], ['x', 'b'], ['p'])  # compute (pusher_center_coordinate) from state variables and slider geometry
        self.p = cs.Function('p', [self.x, self.beta], [self.p_(self.x, self.beta)], ['x', 'b'], ['p'])
        #  -------------------------------------------------------------------
        # pusher position
        self.ctact_ = cs.Function('ctact_', [__x,__beta], [__ctact], ['x', 'b'], ['ctact'])
        self.ctact = cs.Function('ctact', [__x,__beta], [self.ctact_(self.x, self.beta)], ['x', 'b'], ['ctact'])
        # Left contact point position
        self.Ltact_ = cs.Function('Ltact_', [__x,__beta], [__Ltact], ['x', 'b'], ['Ltact'])
        self.Ltact = cs.Function('Ltact', [__x,__beta], [self.Ltact_(self.x, self.beta)], ['x', 'b'], ['Ltact'])
        # Right contact point position
        self.Rtact_ = cs.Function('Rtact_', [__x,__beta], [__Rtact], ['x', 'b'], ['Rtact'])
        self.Rtact = cs.Function('Rtact', [__x,__beta], [self.Rtact_(self.x, self.beta)], ['x', 'b'], ['Rtact'])
        #  -------------------------------------------------------------------
        self.p_ctact_ = cs.Function('p_ctact_', [__x,__beta], [__p_ctact], ['x', 'b'], ['p_ctact'])  # compute (pusher_center_coordinate) from state variables and slider geometry
        self.p_ctact = cs.Function('p_ctact', [self.x, self.beta], [self.p_ctact_(self.x, self.beta)], ['x', 'b'], ['p_ctact'])
        #  -------------------------------------------------------------------
        self.s1 = cs.Function('s1', [self.x], [self.x[0:3]], ['x'], ['s1'])  # compute (x1, y1, θ1) from state variables
        self.s2 = cs.Function('s2', [self.x], [self.x[4:7]], ['x'], ['s2'])  # compute (x2, y2, θ2) from state variables
        #  -------------------------------------------------------------------
        
        # dynamics
        __Jc, __JL, __JR = cs.SX(2,3), cs.SX(2,3), cs.SX(2,3)
        __Jc[0,0] = 1; __Jc[1,1] = 1; __Jc[0,2] = -__yc; __Jc[1,2] = __xc;  # pusher contact jacobian
        __JL[0,0] = -1; __JL[1,1] = -1; __JL[0,2] = __yL; __JL[1,2] = -__xL;  # left contact point contact jacobian
        __JR[0,0] = -1; __JR[1,1] = -1; __JR[0,2] = __yR; __JR[1,2] = -__xR;  # right contact point contact jacobian

        self.RAJc = cs.Function('RAJc', [__x,__beta], [cs.mtimes(cs.mtimes(__R, __A), __Jc.T)], ['x', 'b'], ['RAJc'])
        self.RAJL = cs.Function('RAJL', [__x,__beta], [cs.mtimes(cs.mtimes(__R, __A), __JL.T)], ['x', 'b'], ['RAJL'])
        self.RAJR = cs.Function('RAJR', [__x,__beta], [cs.mtimes(cs.mtimes(__R, __A), __JR.T)], ['x', 'b'], ['RAJR'])
        # the state transition function of slider1
        __f1 = cs.SX(cs.vertcat(cs.mtimes(cs.mtimes(__R,__A),cs.mtimes(__Jc.T,__u[0:2])+cs.mtimes(__JL.T,__u[3:5])+cs.mtimes(__JR.T,__u[5:7])),__u[2]))
        #  -------------------------------------------------------------------
        self.f1_ = cs.Function('f1_', [__x,__u,__beta], [__f1], ['x', 'u', 'b'], ['f1'])  # compute (f1(x, u)) from state variables, input variables and slider geometry
        #  -------------------------------------------------------------------

        # numerize the contact positions to add linear constraints
        x_pseudo = [0., 0., 0., 0., 0., 0., 0., self.psi_r_lim]  # pseudo state variable with only the last dimension valid
        Ltact = self.Ltact_(x_pseudo, self.beta_eval).toarray().squeeze()
        Rtact = self.Rtact_(x_pseudo, self.beta_eval).toarray().squeeze()
        xL, yL = Ltact[0], Ltact[1]
        xR, yR = Rtact[0], Rtact[1]
        xl = self.beta_eval[0]
        xc = -xl/2; yc = -(xl/2)*cs.tan(self.x[3])

        # control constraints
        #  -------------------------------------------------------------------
        try:
            assert self.mode == 'sliding_cc_slack'
            # complementary constraint + slack variables
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            # u[2] - rel sliding vel between pusher and slider counterclockwise
            # u[3] - rel sliding vel between pusher and slider clockwise
            # u[4] - normal force of the left contact point in the local frame
            # u[5] - tangential force of the left contact point in the local frame
            # u[6] - normal force of the right contact point in the local frame
            # u[7] - tangential of the right contact point force in the local frame
            self.Nu = 8  # number of action variables
            self.u = cs.SX.sym('u', self.Nu)
            self.Nz = 2
            self.z = cs.SX.sym('z', self.Nz)
            self.z0 = [1.]*self.Nz
            self.lbz = [-cs.inf]*self.Nz
            self.ubz = [cs.inf]*self.Nz
            # discrete extra variable
            self.z_discrete = False
            self.g_u = cs.Function('g_u', [self.x, self.u, self.z, self.beta], [cs.vertcat(
                # pusher friction cone edges
                self.miu*self.u[0]+self.u[1],
                self.miu*self.u[0]-self.u[1],
                # complementarity constraint
                (self.miu * self.u[0] - self.u[1])*self.u[3] + self.z[0],
                (self.miu * self.u[0] + self.u[1])*self.u[2] + self.z[1],
                # left contact point friction cone edge
                self.miu*self.u[4]+self.u[5],
                self.miu*self.u[4]-self.u[5],
                # right contact point friction cone edge
                self.miu*self.u[6]+self.u[7],
                self.miu*self.u[6]-self.u[7],
                # force balance
                self.u[0]-2*(self.u[4]+self.u[6]),
                self.u[1]-2*(self.u[5]+self.u[7]),
                # torque balance
                (xc*self.u[1]-yc*self.u[0]-xL*self.u[5]+yL*self.u[4]-xR*self.u[7]+yR*self.u[6])-\
                    (-xR*self.u[5]+yR*self.u[4]-xL*self.u[7]+yL*self.u[6])
            )], ['x', 'u', 'other', 'b'], ['g'])
            self.g_lb = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            self.g_ub = [cs.inf, cs.inf, 0., 0., cs.inf, cs.inf, cs.inf, cs.inf, 0., 0., 0.]
            self.Ng_u = 11
            # cost gain for extra variable
            __Ks_max = self.Kz_max
            __Ks_min = self.Kz_min
            __i_th = cs.SX.sym('__i_th')
            self.kz_f = cs.Function('ks', [__i_th], [__Ks_max * cs.exp(__i_th * cs.log(__Ks_min / __Ks_max))])
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim, -cs.inf, -cs.inf, -cs.inf, self.psi_r_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim, cs.inf, cs.inf, cs.inf, self.psi_r_lim]
            self.lbu = [0.0, -self.f_lim, 0.0, 0.0, 0.0, -self.f_lim, 0.0, -self.f_lim]
            self.ubu = [self.f_lim, self.f_lim, self.psi_dot_lim, self.psi_dot_lim, self.f_lim, self.f_lim, self.f_lim, self.f_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f1 = cs.Function('f1', [self.x, self.u, self.beta], [self.f1_(self.x, cs.vertcat(self.u[0:2], self.u[2]-self.u[3], self.u[4:8]), self.beta)],  ['x', 'u', 'b'], ['f1'])
        except:
            print('Multiple sliders case should be launched with  ``{}`` mode!'.format('sliding_cc_slack'))
            sys.exit(-1)
        #  -------------------------------------------------------------------

    def set_patches(self, ax, x_data, beta, vis_flatness=False, u_data=None):
        """
        :param vis_flatness: visualize auxiliary lines for differential flatness or not
        """
        Xl = beta[0]
        Yl = beta[1]
        R_pusher = beta[2]
        x0 = x_data[:, 0]
        # R0 = np.array(self.R(x0))
        R0 = np.eye(3)
        d0 = R0.dot(np.array([-Xl/2., -Yl/2., 0]))
        self.slider = patches.Rectangle(
                x0[0:2]+d0[0:2], Xl, Yl, angle=0.0)
        self.pusher = patches.Circle(
                np.array(self.p(x0, beta)), radius=R_pusher, color='black')
        self.path_past, = ax.plot(x0[0], x0[1], color='orange')
        self.path_future, = ax.plot(x0[0], x0[1],
                color='orange', linestyle='dashed')
        self.cor = patches.Circle(
                np.array([0, 0]), radius=0.002, color='deepskyblue')
        ax.add_patch(self.slider)
        ax.add_patch(self.pusher)
        ax.add_patch(self.cor)
        self.path_past.set_linewidth(2)

        # Set auxiliary lines
        if vis_flatness:
            # set the lines
            self.mid_axis, = ax.plot(0., 0., color='violet', linestyle='dashed')
            self.flat_line, = ax.plot(0., 0., color='violet', linestyle='dashed')
            self.othog_line, = ax.plot(0., 0., color='violet', linestyle='dashed')
            self.force_line, = ax.plot(0., 0., color='violet', linestyle='dashed')
            self.mid_axis.set_linewidth(2.0)
            self.flat_line.set_linewidth(2.0)
            self.othog_line.set_linewidth(2.0)
            self.force_line.set_linewidth(2.0)

            # self.COG = ax.scatter(0, 0, marker='o')

    def animate(self, i, ax, x_data, beta, vis_flatness=False, u_data=None, X_future=None):
        Xl = beta[0]
        Yl = beta[1]
        xi = x_data[:, i]
        # distance between centre of square reference corner
        Ri = np.array(self.R(xi))
        di = Ri.dot(np.array([-Xl/2, -Yl/2, 0]))
        # square reference corner
        ci = xi[0:3] + di
        # compute transformation with respect to rotation angle xi[2]
        trans_ax = ax.transData
        coords = trans_ax.transform(ci[0:2])
        trans_i = transforms.Affine2D().rotate_around(
                coords[0], coords[1], xi[2])
        # Set changes
        self.slider.set_transform(trans_ax+trans_i)
        self.slider.set_xy([ci[0], ci[1]])
        self.pusher.set_center(np.array(self.p(xi, beta)))
        # Set path changes
        if self.path_past is not None:
            self.path_past.set_data(x_data[0, 0:i], x_data[1, 0:i])
        if (self.path_future is not None) and (X_future is not None):
            self.path_future.set_data(X_future[0, :, i], X_future[1, :, i])

        # Set auxiliary lines
        if vis_flatness:
            # Set auxiliary lines
            ab_ratio = (self.A(beta)[0, 0] / self.A(beta)[2, 2]).toarray().squeeze()
            ui = u_data[:, i]
            fn = ui[0]; ft = ui[1]
            ctact_pt = self.ctact_(xi, beta).toarray().squeeze()
            xc = ctact_pt[0]; yc = ctact_pt[1]  # contact point
            r = np.sqrt(xc ** 2 + yc ** 2)
            xt = -ab_ratio * xc / (r ** 2); yt = -ab_ratio * yc / (r ** 2)  # the intersection of central axis and differential flatness line
            # general form of line: Ax + By = C
            # parameter [A, B] of general formalized auxiliary line
            param_AB = np.array([[xc, yc],
                                 [yc, -xc],
                                 [fn, ft],
                                 [ft, -fn]])
            # parameter [C] of general formalized auxiliary line
            param_C = np.array([[xt * xc + yt * yc],
                                [0],
                                [0],
                                [xc * ft - yc * fn]])
            # key points
            Ri_xy = Ri[:2, :2]
            cor_pt = Ri_xy @ np.linalg.inv(param_AB[[0, 2], :]) @ param_C[[0, 2], :].squeeze() + xi[:2] # center of rotation (COR)
            force_pt = Ri_xy @ np.linalg.inv(param_AB[[2, 3], :]) @ param_C[[2, 3], :].squeeze() + xi[:2]  # point on the line of force
            ctact_pt = Ri_xy @ np.array([xc, yc]) + xi[:2]  # contact point
            tilde_pt = Ri_xy @ np.array([xt, yt]) + xi[:2]  # the intersection of central axis and differential flatness line
            # set the lines
            self.mid_axis.set_data([ctact_pt[0], tilde_pt[0]], [ctact_pt[1], tilde_pt[1]])
            self.flat_line.set_data([tilde_pt[0], cor_pt[0]], [tilde_pt[1], cor_pt[1]])
            self.othog_line.set_data([force_pt[0], cor_pt[0]], [force_pt[1], cor_pt[1]])
            self.force_line.set_data([force_pt[0], ctact_pt[0]], [force_pt[1], ctact_pt[1]])
            self.cor.set_center(np.array([cor_pt[0], cor_pt[1]]))
            print(cor_pt)
            # # set the slider's center
            # self.COG.set_xy(xi[0], xi[1])
        return []
    #  -------------------------------------------------------------------

# -------------------------------------------------------------------
