# Author: Yongpeng Jiang
# Date: 17/11/2022
#  -------------------------------------------------------------------
# Description:
#  This script implements a simulation base on limit surface and LCP
#  modeling for simulating double sliders.
#  -------------------------------------------------------------------

#  import libraries
#  -------------------------------------------------------------------
import argparse
from enum import Enum
import numpy as np
import casadi as cs
from copy import deepcopy
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import Polygon
from shapely import ops, affinity
#  -------------------------------------------------------------------
import matlab.engine as engine
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

class Method(Enum):
       qpoases = 'qpoases'
       matlab = 'matlab'

       def __str__(self):
              return self.value
          
def get_contact_point(beta, poseA, poseB):
    """
    Get the nearest point P between poseA and poseB.
    Compute P's coordinate's on poseA in world frame.
    """
    # convert cs.DM to ndarray
    if type(beta) is not np.ndarray:
        beta = beta.toarray().squeeze()
    if type(poseA) is not np.ndarray:
        poseA = poseA.toarray().squeeze()
    if type(poseB) is not np.ndarray:
        poseB = poseB.toarray().squeeze()
        
    x, y = beta[0]/2, beta[1]/2
    base_poly = Polygon([(x, y), (-x, y), (-x, -y), (x, -y), (x, y)])
    polyA = affinity.rotate(base_poly, poseA[2], origin='center', use_radians=True)
    polyB = affinity.rotate(base_poly, poseB[2], origin='center', use_radians=True)
    polyA = affinity.translate(polyA, poseA[0], poseA[1])
    polyB = affinity.translate(polyB, poseB[0], poseB[1])
    pA, _ = ops.nearest_points(polyA, polyB)

    return [pA.x, pA.y]

def rotation_matrix2X2(theta):
    """
    Rotate anti-clockwise
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
#  -------------------------------------------------------------------

# Stacked parameters
#  -------------------------------------------------------------------
# N = 10*8
# dt = 0.005

# theta = [0.5 * np.pi, np.arctan2(0.5*beta[0], 0.5*beta[1])]
# theta = [0.5 * np.pi, 0.25 * np.pi]

# ctact = [[-0.5*beta[0], 0.5*beta[0], -0.5*beta[0]],
#          [          0.,          0., -0.5*beta[1]]]
# ctact = [[-0.5*beta[0], 0.5*beta[0], -0.5*beta[0]],
#          [          0.,       -0.04, -0.5*beta[1]]]

# vp = [0.05, 0.]

# dyn = sliding_pack.dyn_mc.Double_Sys_sq_slider_quasi_static_ellip_lim_surf(
#     planning_config['dynamics'],
#     contactNum=2
# )
#  -------------------------------------------------------------------

# Stacked code
#  -------------------------------------------------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument('-sa', '--show_anim', action='store_true', default=True, help='show animation')
# parser.add_argument('-sm', '--save_mp4', action='store_true', default=True, help='save the animation as mp4')
# parser.add_argument('-m', '--method', type=Method, choices=list(Method), help='LCP solver (qpoases or matlab)', required=True)
# args = parser.parse_args()
#  -------------------------------------------------------------------
# if args.method == Method.matlab:
#     eng = engine.start_matlab()
#  -------------------------------------------------------------------

# Get config files
#  -------------------------------------------------------------------
# planning_config = sliding_pack.load_config('planning_config.yaml')
#  -------------------------------------------------------------------

# beta, theta, ctact, vp = cs.DM(beta), cs.DM(theta), cs.DM(ctact), cs.DM(vp)

"""
Q = matrix((2*M).toarray())
p = matrix(q.toarray().squeeze())
G = np.zeros((16, 8))
G[:8, :] = -M.toarray()
G[8:, :] = -np.ones(8)
G = matrix(G)
h = np.zeros(16)
h[:8] = q.toarray().squeeze()
h = matrix(h)
A = matrix(np.zeros((1, 8)))
b = matrix(0.)

sol = solvers.qp(P=Q, q=p, G=G, h=h, A=A, b=b)
"""

# x_a0, x_b0 = np.array([0, 0, 0.5*np.pi]), np.array([-0.01767767, 0.10217514, 0.25*np.pi])
# x_a0, x_b0 = np.array([0, 0, 0.5*np.pi]), np.array([-0.04889688, 0.08483655, (75./180.)*np.pi])
# x_a0, x_b0 = np.array([0, 0, 0.5*np.pi]), np.array([-0.00889688, 0.08483655, (75./180.)*np.pi])
#  -------------------------------------------------------------------

# Simulation object
#  -------------------------------------------------------------------
class buildDoubleSliderSimObj():
    
    def __init__(self, dyn_class, timeHorizon, configDict, dt=0.01, method='qpoases',
                 showAnimFlag=True, saveFileFlag=False) -> None:
        # simulation constants
        self.N = timeHorizon
        self.dt = dt
        self.configDict = configDict
        self.method = method
        self.showAnimFlag = showAnimFlag
        self.saveFileFlag = saveFileFlag
        self.pusherVelLim = configDict['pusherVelLim']

        self.beta = [
            self.configDict['xLenght'],
            self.configDict['yLenght'],
            self.configDict['pusherRadious']
        ]
        
        self.db_dyn = dyn_class
        
        if self.method == 'matlab':
            self.eng = engine.start_matlab()
            
    def run_one_step(self, x_a, x_b, ctact, vp, beta, printFlag=False):
        """
        Simulate one timestep.
        :param x_a: SliderA's current pose.
        :param x_b: SliderB's current pose.
        :param ctact: SliderA & B & Pusher's current contact locations.
        :param vp: Pusher velocity.
        :param beta: Slider's contact geometry.
        :return x_a_new: SliderA's next pose.
        :return x_b_new: SliderB's next pose.
        :return ctact_new: SliderA & B & Pusher's new contact locations.
        """
        # azimuth angle
        theta = cs.horzcat(x_a[-1], x_b[-1])
        
        vp[0] = max(-self.pusherVelLim, min(self.pusherVelLim, vp[0]))
        vp[1] = max(-self.pusherVelLim, min(self.pusherVelLim, vp[1]))
        
        # matrices for optimization
        w = self.db_dyn.w
        z = self.db_dyn.z
        M = self.db_dyn.M(beta, theta, ctact.T)
        q = self.db_dyn.q(vp)
        
        # create quadratic programming to solve the LCP
        if self.method == 'qpoases':
            qp = {}
            qp['h'] = (2*M).sparsity()
            qp['a'] = M.sparsity()
            S = cs.conic('LCP2QP', 'qpoases', qp)

            r = S(h=2*M, g=q, a=M, lba=-q, lbx=0.)
            z_ctact = r['x']
        elif self.method == 'matlab':
            ans = self.eng.LCP(M.toarray(), q.toarray().reshape(8,1))
            z_ctact = cs.DM(np.array(ans).squeeze())
        else:
            NotImplementedError('The method requested was not implemented try qpoases or matlab solver!')
        
        slack = M @ z_ctact + q
        
        # report solution details
        z_ctact_ = z_ctact.toarray().squeeze()
        if printFlag:
            print('force: ', [z_ctact_[0], z_ctact_[2]-z_ctact_[3]], [z_ctact_[1], z_ctact_[4]-z_ctact_[5]])
            print('wrenchA: ', self.db_dyn.wrenchA(beta,theta,ctact.T,z_ctact).toarray().squeeze())
            print('wrenchB: ', self.db_dyn.wrenchB(beta,theta,ctact.T,z_ctact).toarray().squeeze())
            print('twistA: ', self.db_dyn.VA(beta,theta,ctact.T,z_ctact).toarray().squeeze())
            print('twistB: ', self.db_dyn.VB(beta,theta,ctact.T,z_ctact).toarray().squeeze())
            if self.method == 'qpoases':
                print('cost: ', r['cost'])
            print('alpha: ', slack[:2])
            print('beta: ', slack[2:6])
            print('gamma: ', slack[6:])
            print('lamda: ', z_ctact[6:])
        
        # update state variables
        x_a_new = x_a + self.dt*self.db_dyn.fA(beta, theta, ctact.T, z_ctact)
        x_b_new = x_b + self.dt*self.db_dyn.fB(beta, theta, ctact.T, z_ctact)  # since dt -\-> 0, penetration between slider A and B might occur
        
        # update the contact point between two sliders
        # ab_ctact_pt = get_contact_point(beta, x_a_new, x_b_new)  # penetration might occur
        ctact_new = deepcopy(ctact)
        # update the push point
        ctact_new[:, 0] = ctact[:, 0] + self.dt*self.db_dyn.fvp(beta, theta, ctact.T, z_ctact, vp)
        # update the slider's contact point
        ctact_new[:, 1] = ctact[:, 1] + self.dt*self.db_dyn.vpB(beta, theta, ctact.T, z_ctact)
        R_a = self.db_dyn.RA(x_a_new[2])
        R_b = self.db_dyn.RB(x_b_new[2])
        x_b_new[:2] = (x_a_new[:2] + R_a[:2, :2] @ ctact_new[:, 1]) - R_b[:2, :2] @ np.array([-beta[0]/2, -beta[1]/2]).squeeze()  # no penetration is guaranteed

        return x_a_new, x_b_new, ctact_new
        
    def run(self, x_init=None, u_opt=None):
        # initialize simulation variables
        #  -------------------------------------------------------------------
        beta = cs.DM(self.beta)
        theta = cs.DM([0.5*np.pi, 0.5*np.pi+x_init[2]])
        ctact = cs.DM([[-0.5*self.beta[0], 0.5*self.beta[0], -0.5*self.beta[0]],
                       [        x_init[0],        x_init[1], -0.5*self.beta[1]]])
        vp = u_opt if type(u_opt) == cs.DM else cs.DM(u_opt)
        
        # initialize slider states
        #  -------------------------------------------------------------------
        x_a0 = np.array([0.0, 0.0, 0.5*np.pi])
        x_b0 = np.zeros_like(x_a0)
        x_b0[:2] = x_a0[:2] + rotation_matrix2X2(0.5*np.pi) @ np.array(ctact[:, 1]).squeeze().astype(np.float32) - \
                   rotation_matrix2X2(float(x_init[2])) @ np.array([-0.5*self.beta[0], -0.5*self.beta[1]])
        x_b0[2] = x_a0[2] + float(x_init[2])
                   
        X_A, X_B = np.expand_dims(x_a0, axis=0), np.expand_dims(x_b0, axis=0)
        X_ctact = np.expand_dims(ctact[:, 0].toarray().squeeze(), axis=0)  # the pusher contact point on slider A

        n_iter = 0
        # run simulation step by step
        #  -------------------------------------------------------------------
        while True:
            x_a, x_b = X_A[-1, :], X_B[-1, :]
            
            print('---------- iters:{0} ----------'.format(n_iter))
            x_a_new, x_b_new, ctact = self.run_one_step(x_a, x_b, ctact, vp[:, n_iter], beta, printFlag=False)
            
            # update the azimuth angle
            theta[0] = x_a_new[-1]
            theta[1] = x_b_new[-1]
            
            # append new states
            X_A = np.concatenate((X_A, x_a_new.T), axis=0)
            X_B = np.concatenate((X_B, x_b_new.T), axis=0)
            X_ctact = np.concatenate((X_ctact, np.expand_dims(ctact[:, 0].toarray().squeeze(), axis=0)), axis=0)
            
            n_iter += 1
            if n_iter >= self.N:
                break
        return X_A, X_B, X_ctact
    
    def visualization(self, X_A, X_B, X_ctact, beta):
        # plot simulation and save file
        #  -------------------------------------------------------------------
        plt.rcParams['figure.dpi'] = 150
        if self.showAnimFlag:
            #  ---------------------------------------------------------------
            # plot settings
            window_title = 'Simulation Result'
            plot_title = 'Single Pusher Double Sliders'
            
            X_A, X_B = X_A.T, X_B.T
            X_ctact = X_ctact.T
            beta = beta.toarray().squeeze().tolist()
            
            fig, ax = plt.subplots()
            fig.canvas.set_window_title(window_title)
            ax.set_xlim([-0.2, 0.5])
            ax.set_ylim([-0.2, 0.5])
            ax.set_autoscale_on(False)
            ax.grid()
            ax.set_aspect('equal', 'box')
            ax.set_title(plot_title)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            fig.set_size_inches(8, 6, forward=True)
            
            # set patches
            self.db_dyn.set_patches(ax, X_A, X_B, X_ctact, beta)
            # show animation
            ani = animation.FuncAnimation(
                    fig,
                    self.db_dyn.animate,
                    fargs=(ax, X_A, X_B, X_ctact, beta),
                    frames=X_ctact.shape[1]-1,
                    interval=self.dt*1000*5,  # microseconds
                    blit=True,
                    repeat=False,
            )
            
            if self.saveFileFlag:
                ani.save('./video/simu_align_sliders_mpc.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
            
            plt.show()
            
            
if __name__ == '__main__':
    # simulation constant
    N = 10*8
    dt = 0.005
    x_init = [-0.04, -0.04, (75./180.)*np.pi-0.5*np.pi]
    u_opt = np.array([0.05, 0.]).reshape(-1, 1).repeat(N, axis=1)

    # load config dict
    planning_config = sliding_pack.load_config('planning_config.yaml')
    
    # build dynamics
    dyn = sliding_pack.dyn_mc.Double_Sys_sq_slider_quasi_static_ellip_lim_surf(
        planning_config['dynamics'],
        contactNum=2
    )
    
    # build simulation object
    simObj = buildDoubleSliderSimObj(
        dyn, N, planning_config['dynamics'], dt, method='qpoases',
        showAnimFlag=True, saveFileFlag=False)
    
    # run simulation
    simObj.run(x_init, u_opt)
