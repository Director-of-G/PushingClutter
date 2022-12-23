# Author: Yongpeng Jiang
# Date: 21/12/2022
#  -------------------------------------------------------------------
# Description:
#  This script implements a non-linear program (NLP) model predictive
#  controller (MPC) for controlling relative movements between two sliders.
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

def clip(num, minval, maxval):
    return max(minval, min(maxval, num))

# Get config files
#  -------------------------------------------------------------------
dbplan_slider_config = sliding_pack.load_config('planning_config_2sliders.yaml')
dbtrack_slider_config = sliding_pack.load_config('tracking_config_2sliders.yaml')
#  -------------------------------------------------------------------
          
# Set Problem constants
#  -------------------------------------------------------------------
T = 2.5  # time of the simulation is seconds
freq = 60  # number of increments per second
show_anim = True
save_to_file = True
#  -------------------------------------------------------------------
# Compute Problem constants
#  -------------------------------------------------------------------
dt = 1.0/freq  # sampling time
N = int(T*freq)  # total number of iterations
N_MPC = 20
#  -------------------------------------------------------------------

# define system dynamics
#  -------------------------------------------------------------------
dbplan_dyn = sliding_pack.dyn_mc.Double_Sys_sq_slider_quasi_static_ellip_lim_surf(
    dbplan_slider_config['dynamics'],
    contactNum=2,
    controlRelPose=False
)

dbtrack_dyn = sliding_pack.dyn_mc.Double_Sys_sq_slider_quasi_static_ellip_lim_surf(
    dbtrack_slider_config['dynamics'],
    contactNum=2,
    controlRelPose=False
)

#  ------------------------------------------------------------------
beta = [
    dbplan_slider_config['dynamics']['xLenght'],
    dbplan_slider_config['dynamics']['yLenght'],
    dbplan_slider_config['dynamics']['pusherRadious']
]
thetaA0 = 0.5*np.pi
thetaB0 = (45./180.)*np.pi
dtheta0 = thetaB0-thetaA0

# x_init = [0., -0.25*beta[1], dtheta0]  # control relative motions
x_init = [0., 0.25*beta[1], dtheta0, 0., 0., 0.5*np.pi]  # control sliderA pose

# Compute nominal actions for sticking contact
#  ------------------------------------------------------------------
# X_nom_val = np.zeros((3, N))
# X_nom_val[1, :] = np.linspace(x_init[1], -0.5*beta[1], N)
# X_nom_val[2, :] = np.linspace(x_init[2], 0., N)
# X_nom_val = np.concatenate((X_nom_val, X_nom_val[:, -1].reshape(-1, 1)+(X_nom_val-X_nom_val[:, 0].reshape(-1, 1))[:, 1:N_MPC+1]), axis=1)
X_nom_val = sliding_pack.traj.generate_traj_align_sliders(x_init[1], -0.5*beta[1], x_init[2], N, N_MPC)
# U_nom_val = np.zeros((2, N))
# U_nom_val[0, :] = 0.0
# U_nom_val[1, :] = 0.0
# U_nom_val = np.concatenate((U_nom_val, U_nom_val[:, 1:N_MPC+1]), axis=1)
U_nom_val = np.zeros((2, N+N_MPC-1))

# Nominal trajectory to control only sliderA pose
X_goal = [0., 0.09, 0.5*np.pi]
x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(X_goal[0], X_goal[1], N, N_MPC)
X_nom_val_aug, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
X_nom_val = np.concatenate((X_nom_val, X_nom_val_aug[:3, :]), axis=0)

# Build the planning optimization
#  ------------------------------------------------------------------
import pdb; pdb.set_trace()
dbplan_optObj = sliding_pack.db_to.buildDoubleSliderOptObj(
    dbplan_dyn, N+N_MPC, dbplan_slider_config['TO'], X_nom_val=X_nom_val, dt=dt, maxIter=3000, controlRelPose=False)

#  ------------------------------------------------------------------

# Solve the nonlinear optimization.
#  ------------------------------------------------------------------
# Set x_warmstart to be the first point.
# resultFlag, x_opt, u_opt, z_opt, w_opt, other_opt, f_opt, t_opt = dbplan_optObj.solveProblem(
#     0, x_init, beta, X_warmStart=X_nom_val[:, 0].reshape(-1, 1).repeat(N+N_MPC, 1), U_warmStart=U_nom_val)

# Set x_warmstart to be a whole trajectory.
#  ------------------------------------------------------------------
resultFlag, x_opt, u_opt, z_opt, w_opt, other_opt, f_opt, t_opt = dbplan_optObj.solveProblem(
    0, x_init, beta, X_warmStart=X_nom_val, U_warmStart=U_nom_val)

import pdb; pdb.set_trace()
# # Plot Optimization Results
# #  -------------------------------------------------------------------
fig, axs = plt.subplots(3, 4, sharex=True)
fig.set_size_inches(10, 10, forward=True)
t_Nx = np.linspace(0, T, N)
t_Nu = np.linspace(0, T, N-1)
# #  -------------------------------------------------------------------
# # plot contact position
for i in [0, 1]:
    axs[0, 0].plot(t_Nx, x_opt[i, 0:N].T, linestyle='--', label='x{0}'.format(i))
handles, labels = axs[0, 0].get_legend_handles_labels()
axs[0, 0].legend(handles, labels)
axs[0, 0].set_xlabel('time [s]')
axs[0, 0].set_ylabel('y [m]')
axs[0, 0].grid()
# #  -------------------------------------------------------------------
# # plot position
for i in [3, 4]:
    axs[0, 1].plot(t_Nx, x_opt[i, 0:N].T, linestyle='--', label='x' if i == 3 else 'y')
    axs[0, 1].plot(t_Nx, X_nom_val[i, 0:N].T, linestyle='--', label='x_nom' if i == 3 else 'y_nom')
handles, labels = axs[0, 1].get_legend_handles_labels()
axs[0, 1].legend(handles, labels)
axs[0, 1].set_xlabel('time [s]')
axs[0, 1].set_ylabel('x or y [m]')
axs[0, 1].grid()
# #  -------------------------------------------------------------------
# # plot azimuth angle
for i in [2, 5]:
    axs[0, 2].plot(t_Nx, x_opt[i, 0:N].T, linestyle='--', label='dtheta' if i == 2 else 'theta')
    if i == 5:
        axs[0, 2].plot(t_Nx, X_nom_val[i, 0:N].T, linestyle='--', label='theta_nom')
handles, labels = axs[0, 2].get_legend_handles_labels()
axs[0, 2].legend(handles, labels)
axs[0, 2].set_xlabel('time [s]')
axs[0, 2].set_ylabel('[rad]')
axs[0, 2].grid()
# #  -------------------------------------------------------------------
# # plot input
for i in [0, 1]:
    axs[0, 3].plot(t_Nu, u_opt[i, 0:N-1].T, linestyle='--', label='u{0}'.format(i))
handles, labels = axs[0, 3].get_legend_handles_labels()
axs[0, 3].legend(handles, labels)
axs[0, 3].set_xlabel('time [s]')
axs[0, 3].set_ylabel('[m/2]')
axs[0, 3].grid()
# #  -------------------------------------------------------------------
# # plot auxiliary variables (force)
axs[1, 0].plot(t_Nu, z_opt[0, 0:N-1].T, linestyle='--', label='f_n0')
axs[1, 0].plot(t_Nu, z_opt[2, 0:N-1].T-z_opt[3, 0:N-1].T, linestyle='--', label='f_t0')
handles, labels = axs[1, 0].get_legend_handles_labels()
axs[1, 0].legend(handles, labels)
axs[1, 0].set_xlabel('time [s]')
axs[1, 0].set_ylabel('[N]')
axs[1, 0].grid()

axs[1, 1].plot(t_Nu, z_opt[1, 0:N-1].T, linestyle='--', label='f_n1')
axs[1, 1].plot(t_Nu, z_opt[4, 0:N-1].T-z_opt[5, 0:N-1].T, linestyle='--', label='f_t1')
handles, labels = axs[1, 1].get_legend_handles_labels()
axs[1, 1].legend(handles, labels)
axs[1, 1].set_xlabel('time [s]')
axs[1, 1].set_ylabel('[N]')
axs[1, 1].grid()
# #  -------------------------------------------------------------------
# # plot auxiliary variables (mu and gamma)
for i in [6, 7]:
    axs[1, 2].plot(t_Nu, z_opt[i, 0:N-1].T, linestyle='--', label='z{0}'.format(i))
    axs[1, 2].plot(t_Nu, w_opt[i, 0:N-1].T, linestyle='--', label='w{0}'.format(i))
handles, labels = axs[1, 2].get_legend_handles_labels()
axs[1, 2].legend(handles, labels)
axs[1, 2].set_xlabel('time [s]')
axs[1, 2].set_ylabel('z or w')
axs[1, 2].grid()

for i in [0, 1]:
    axs[2, 0].plot(t_Nu, z_opt[i, 0:N-1].T, linestyle='--', label='z{0}'.format(i))
    axs[2, 0].plot(t_Nu, w_opt[i, 0:N-1].T, linestyle='--', label='w{0}'.format(i))
handles, labels = axs[2, 0].get_legend_handles_labels()
axs[2, 0].legend(handles, labels)
axs[2, 0].set_xlabel('time [s]')
axs[2, 0].set_ylabel('z or w')
axs[2, 0].grid()

for i in [2, 3]:
    axs[2, 1].plot(t_Nu, z_opt[i, 0:N-1].T, linestyle='--', label='z{0}'.format(i))
    axs[2, 1].plot(t_Nu, w_opt[i, 0:N-1].T, linestyle='--', label='w{0}'.format(i))
handles, labels = axs[2, 1].get_legend_handles_labels()
axs[2, 1].legend(handles, labels)
axs[2, 1].set_xlabel('time [s]')
axs[2, 1].set_ylabel('z or w')
axs[2, 1].grid()

for i in [4, 5]:
    axs[2, 2].plot(t_Nu, z_opt[i, 0:N-1].T, linestyle='--', label='z{0}'.format(i))
    axs[2, 2].plot(t_Nu, w_opt[i, 0:N-1].T, linestyle='--', label='w{0}'.format(i))
handles, labels = axs[2, 2].get_legend_handles_labels()
axs[2, 2].legend(handles, labels)
axs[2, 2].set_xlabel('time [s]')
axs[2, 2].set_ylabel('z or w')
axs[2, 2].grid()
# #  -------------------------------------------------------------------
# # plot slack variables (mu and gamma)
for i in [0, 1, 2, 3]:
    axs[1, 3].plot(t_Nu, other_opt[i, 0:N-1].T, linestyle='--', label='s{0}'.format(i))
handles, labels = axs[1, 3].get_legend_handles_labels()
axs[1, 3].legend(handles, labels)
axs[1, 3].set_xlabel('time [s]')
axs[1, 3].set_ylabel('s')
axs[1, 3].grid()

for i in [4, 5, 6, 7]:
    axs[2, 3].plot(t_Nu, other_opt[i, 0:N-1].T, linestyle='--', label='s{0}'.format(i))
handles, labels = axs[2, 3].get_legend_handles_labels()
axs[2, 3].legend(handles, labels)
axs[2, 3].set_xlabel('time [s]')
axs[2, 3].set_ylabel('s')
axs[2, 3].grid()
plt.show()
# #  -------------------------------------------------------------------

# Save the marices for debug
# #  -------------------------------------------------------------------
M_opt = np.empty((0, dbplan_dyn.Nw, dbplan_dyn.Nw))
q_opt = np.empty((0, dbplan_dyn.Nw))
for i in range(N-1):
    M_opt = np.concatenate((M_opt, np.expand_dims(dbplan_dyn.M_opt(beta, x_opt[:, i]).toarray().squeeze(), axis=0)), axis=0)
    q_opt = np.concatenate((q_opt, np.expand_dims(dbplan_dyn.q_opt(u_opt[:, i]).toarray().squeeze(), axis=0)), axis=0)
np.save('./data/M_opt.npy', M_opt)
np.save('./data/q_opt.npy', q_opt)
np.save('./data/w_opt', w_opt)
np.save('./data/z_opt', z_opt)
np.save('./data/u_opt', u_opt)
np.save('./data/x_opt', x_opt)
    

np.save('u_plan_double_slider.npy', u_opt)

# Build the tracking optimization
#  ------------------------------------------------------------------
x0 = x_init
X_track = np.empty([dbtrack_dyn.Nx, 0])
U_track = np.empty([dbtrack_dyn.Nu, 0])
X_track= np.concatenate((X_track, np.expand_dims(x0, axis=1)), axis=1)

dbtrack_optObj = sliding_pack.db_to.buildDoubleSliderOptObj(
    dbtrack_dyn, N_MPC, dbtrack_slider_config['TO'], X_nom_val=X_nom_val, dt=dt, maxIter=3000, controlRelPose=False)

# Offline tracking does not work, the LCP gives different solutions in control and simulation,
# probably due to different solution precisions.
# for idx in range(int(N)-1):
#     print('-------------------------')
#     print(idx)
#     resultFlag, x_opt, u_opt, z_opt, w_opt, other_opt, f_opt, t_opt = dbtrack_optObj.solveProblem(
#         idx, x0, beta, X_warmStart=X_nom_val[:, 0].reshape(-1, 1).repeat(N+N_MPC, 1), U_warmStart=U_nom_val)
#     print(f_opt)
#     u0 = u_opt[:, 0].elements()
#     z0 = z_opt[:, 0].elements()
#     x0 = (x0 + dt*dbtrack_dyn.f_opt(beta, x0, u0, z0)).elements()
#     X_track[:, idx+1] = x0
#     U_track[:, idx] = u0

# Online tracking.
#  -------------------------------------------------------------------
# initialize simulation variables.
#  -------------------------------------------------------------------
pusherVelLim = dbtrack_slider_config['dynamics']['pusherVelLim']
ctact = cs.DM([[-0.5*beta[0], 0.5*beta[0], -0.5*beta[0]],
               [       x0[0],       x0[1], -0.5*beta[1]]])
beta = cs.DM(beta)
theta = cs.DM([0.5*np.pi, 0.5*np.pi+x0[2]])

# initialize slider states.
#  -------------------------------------------------------------------
import pdb; pdb.set_trace()
x_a0 = np.array([0.0, 0.0, 0.5*np.pi])
x_b0 = np.zeros_like(x_a0)
x_b0[:2] = x_a0[:2] + sliding_pack.db_sim.rotation_matrix2X2(float(theta[0])) @ np.array(ctact[:, 1]).squeeze().astype(np.float32) - \
           sliding_pack.db_sim.rotation_matrix2X2(float(theta[1])) @ np.array(ctact[:, 2]).squeeze().astype(np.float32)
x_b0[2] = float(theta[1])

db_simObj = sliding_pack.db_sim.buildDoubleSliderSimObj(
    dbtrack_dyn, N-1, dbplan_slider_config['dynamics'], dt, method='matlab',
    showAnimFlag=True, saveFileFlag=True
)

X_A = x_opt[3:, :].toarray().squeeze().T
X_B = np.zeros_like(X_A)
X_ctact = np.expand_dims([-0.5*float(beta[0]), float(x_opt[0, 0])], axis=0)
for i in range(X_A.shape[0]):
    x_ai = X_A[i, :]
    x_bi = np.zeros_like(x_ai)
    x_bi[:2] = x_ai[:2] + sliding_pack.db_sim.rotation_matrix2X2(x_ai[2]).squeeze() @ np.array([0.5*float(beta[0]), float(x_opt[1, i])]) - \
                sliding_pack.db_sim.rotation_matrix2X2(x_ai[2]+float(x_opt[2, i])).squeeze() @ np.array([-0.5*float(beta[0]), -0.5*float(beta[1])])
    x_bi[2] = x_ai[2].squeeze() + float(x_opt[2, i])
    X_B[i, :] = x_bi
    X_ctact = np.concatenate((X_ctact, np.expand_dims([-0.5*float(beta[0]), float(x_opt[0, i])], axis=0)), axis=0)
# X_A, X_B, X_ctact = db_simObj.run(x_init, u_opt)
db_simObj.visualization(X_A, X_B, X_ctact, beta)

import pdb; pdb.set_trace()

X_A, X_B, X_ctact = db_simObj.run(x_init, u_opt)
db_simObj.visualization(X_A, X_B, X_ctact, beta)

import pdb; pdb.set_trace()

x_a, x_b = deepcopy(x_a0), deepcopy(x_b0)

# for visualization
X_A, X_B = np.expand_dims(x_a0, axis=0), np.expand_dims(x_b0, axis=0)
X_ctact = np.expand_dims(ctact[:, 0].toarray().squeeze(), axis=0) 
        
n_iters = 0
while True:
    # idx = sliding_pack.traj.compute_nearest_point_index(X_nom_val[1, :], ctact[1, 1])
    idx = sliding_pack.traj.compute_nearest_point_index(X_nom_val[3:5, :], x_a[0:2], ord=2)
    if idx >= int(N)-1 or n_iters >= 250:
        break
    resultFlag, x_opt, u_opt, z_opt, w_opt, other_opt, f_opt, t_opt = dbtrack_optObj.solveProblem(
        idx, x0, beta.toarray().squeeze().tolist(), X_warmStart=X_nom_val, U_warmStart=U_nom_val)
    
    # print('s_opt: ', other_opt.toarray().squeeze())
    
    x_a_new, x_b_new, ctact_new = db_simObj.run_one_step(x_a, x_b, ctact, 10*u_opt[:, 0], beta, printFlag=True)
    x_a, x_b, ctact = deepcopy(x_a_new), deepcopy(x_b_new), deepcopy(ctact_new)
    
    # append new states
    X_A = np.concatenate((X_A, x_a_new.T), axis=0)
    X_B = np.concatenate((X_B, x_b_new.T), axis=0)
    X_ctact = np.concatenate((X_ctact, np.expand_dims(ctact_new[:, 0].toarray().squeeze(), axis=0)), axis=0)
    
    x_new = [float(ctact_new[1, 0]), float(ctact_new[1, 1]), float(x_b_new[-1] - x_a_new[-1]), float(x_a_new[0]), float(x_a_new[1]), float(x_a_new[2])]
    X_track = np.concatenate((X_track, np.expand_dims(x_new, axis=1)), axis=1)
    U_track = np.concatenate((U_track, np.expand_dims(u_opt[:, 0].toarray().squeeze(), axis=1)), axis=1)
    
    x0 = deepcopy(x_new)
    
    print('---- n_iters:{0}, idx:{1} -----'.format(n_iters, idx))
    
    n_iters += 1

np.save('u_track_double_slider.npy', U_track)

u_opt = np.load('u_track_double_slider.npy')
import pdb; pdb.set_trace()

db_simObj.visualization(X_A, X_B, X_ctact, beta)

# u_opt[0, :] = 0.02
# u_opt[1, :] = 0.
# x_init = [-0.4*beta[1], -0.25*beta[1], dtheta0]

# Running simulation and visualize the results
#  ------------------------------------------------------------------
# db_simObj = sliding_pack.db_sim.buildDoubleSliderSimObj(
#     dbplan_dyn, N-1, dbplan_slider_config['dynamics'], dt, x_init, u_opt, method='matlab',
#     showAnimFlag=True, saveFileFlag=False
# )

# db_simObj.run()
