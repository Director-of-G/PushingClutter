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
N_MPC = 15
#  -------------------------------------------------------------------

# define system dynamics
#  -------------------------------------------------------------------
dbplan_dyn = sliding_pack.dyn_mc.Double_Sys_sq_slider_quasi_static_ellip_lim_surf(
    dbplan_slider_config['dynamics'],
    contactNum=2
)

dbtrack_dyn = sliding_pack.dyn_mc.Double_Sys_sq_slider_quasi_static_ellip_lim_surf(
    dbtrack_slider_config['dynamics'],
    contactNum=2
)

#  ------------------------------------------------------------------
beta = [
    dbplan_slider_config['dynamics']['xLenght'],
    dbplan_slider_config['dynamics']['yLenght'],
    dbplan_slider_config['dynamics']['pusherRadious']
]
thetaA0 = 0.5*np.pi
thetaB0 = (75./180.)*np.pi
dtheta0 = thetaB0-thetaA0
x_init = [0., -0.25*beta[1], dtheta0]

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

# Build the planning optimization
#  ------------------------------------------------------------------
dbplan_optObj = sliding_pack.db_to.buildDoubleSliderOptObj(
    dbplan_dyn, N+N_MPC, dbplan_slider_config['TO'], X_nom_val=X_nom_val, dt=dt, maxIter=200)

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


np.save('u_plan_double_slider.npy', u_opt)

# Build the tracking optimization
#  ------------------------------------------------------------------
x0 = x_init
X_track = np.empty([dbtrack_dyn.Nx, 0])
U_track = np.empty([dbtrack_dyn.Nu, 0])
X_track= np.concatenate((X_track, np.expand_dims(x0, axis=1)), axis=1)

dbtrack_optObj = sliding_pack.db_to.buildDoubleSliderOptObj(
    dbtrack_dyn, N_MPC, dbtrack_slider_config['TO'], X_nom_val=x_opt, dt=dt, maxIter=200)

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
x_b0[:2] = x_a0[:2] + sliding_pack.db_sim.rotation_matrix2X2(0.5*np.pi) @ np.array(ctact[:, 1]).squeeze().astype(np.float32) - \
           sliding_pack.db_sim.rotation_matrix2X2(float(x_init[2])) @ np.array([-0.5*float(beta[0]), -0.5*float(beta[1])])
x_b0[2] = x_a0[2] + float(x_init[2])

db_simObj = sliding_pack.db_sim.buildDoubleSliderSimObj(
    dbtrack_dyn, N-1, dbplan_slider_config['dynamics'], dt, method='matlab',
    showAnimFlag=True, saveFileFlag=True
)

x_a, x_b = deepcopy(x_a0), deepcopy(x_b0)

# for visualization
X_A, X_B = np.expand_dims(x_a0, axis=0), np.expand_dims(x_b0, axis=0)
X_ctact = np.expand_dims(ctact[:, 0].toarray().squeeze(), axis=0) 
        
n_iters = 0
while True:
    idx = sliding_pack.traj.compute_nearest_point_index(X_nom_val[1, :], ctact[1, 1])
    if idx >= int(N)-1:
        break
    resultFlag, x_opt, u_opt, z_opt, w_opt, other_opt, f_opt, t_opt = dbtrack_optObj.solveProblem(
        idx, x0, beta.toarray().squeeze().tolist(), X_warmStart=X_nom_val, U_warmStart=U_nom_val)
    
    x_a_new, x_b_new, ctact_new = db_simObj.run_one_step(x_a, x_b, ctact, 10*u_opt[:, 0], beta, printFlag=True)
    x_a, x_b, ctact = deepcopy(x_a_new), deepcopy(x_b_new), deepcopy(ctact_new)
    
    # append new states
    X_A = np.concatenate((X_A, x_a_new.T), axis=0)
    X_B = np.concatenate((X_B, x_b_new.T), axis=0)
    X_ctact = np.concatenate((X_ctact, np.expand_dims(ctact_new[:, 0].toarray().squeeze(), axis=0)), axis=0)
    
    x_new = [float(ctact_new[1, 0]), float(ctact_new[1, 1]), float(x_b_new[-1] - x_a_new[-1])]
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
