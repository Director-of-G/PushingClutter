# Author: Joao Moura
# Date: 21/08/2020
#  -------------------------------------------------------------------
# Description:
#  This script implements a non-linear program (NLP) model predictive controller (MPC)
#  for tracking a trajectory of a square slider object with a single
#  and sliding contact pusher.
#  -------------------------------------------------------------------

#  import libraries
#  -------------------------------------------------------------------
import sys
import yaml
from copy import deepcopy
import numpy as np
import pandas as pd
import casadi as cs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

# Get config files
#  -------------------------------------------------------------------
tracking_config = sliding_pack.load_config('tracking_config.yaml')
planning_config = sliding_pack.load_config('nom_config.yaml')
double_slider_config = sliding_pack.load_config('tracking_config_2sliders.yaml')
#  -------------------------------------------------------------------

# Set Problem constants
#  -------------------------------------------------------------------
T = 10  # time of the simulation is seconds
freq = 25  # number of increments per second
# N_MPC = 12 # time horizon for the MPC controller
N_MPC = 25  # time horizon for the MPC controller
# x_init_val = [-0.03, 0.03, 30*(np.pi/180.), 0]
x_init_val = [0., 0., 90.*(np.pi/180.), 0]
show_anim = True
save_to_file = False
#  -------------------------------------------------------------------
# Computing Problem constants
#  -------------------------------------------------------------------
dt = 1.0/freq  # sampling time
N = int(T*freq)  # total number of iterations
K_d = 2.0  # disturbance observer gain
Nidx = int(N)
idxDist = 5.*freq
# Nidx = 10
#  -------------------------------------------------------------------

# define system dynamics
#  -------------------------------------------------------------------
dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
    tracking_config['dynamics'],
    tracking_config['TO']['contactMode']
)
double_dyn = sliding_pack.dyn_test.Double_Slider_lim_surf_mini_dyn(
    double_slider_config['dynamics']
)
#  -------------------------------------------------------------------

# Generate Nominal Trajectory
#  -------------------------------------------------------------------
X_goal = [0.0, 0.3, (90./180.)*np.pi]
# print(X_goal)
x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(X_goal[0], X_goal[1], N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.5, 0.3, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.2, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_ellipse(-np.pi/2, 3*np.pi/2, 0.2, 0.1, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_eight(0.3, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_sine(0.0, 1.0, 0.5, 0.05, N, N_MPC)
#  -------------------------------------------------------------------
# stack state and derivative of state
X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
#  ------------------------------------------------------------------

# Compute nominal actions for sticking contact
#  ------------------------------------------------------------------
dynNom = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
        planning_config['dynamics'],
        planning_config['TO']['contactMode']
)
optObjNom = sliding_pack.to.buildOptObj(
        dynNom, N+N_MPC, planning_config['TO'], dt=dt)
beta = [
    planning_config['dynamics']['xLenght'],
    planning_config['dynamics']['yLenght'],
    planning_config['dynamics']['pusherRadious']
]

# Configure the Obstacle Slider
#  -------------------------------------------------------------------
dtheta = -(5./180.)*np.pi
ctact_config = [-beta[1]/2, beta[1]/6]  # yA1, yB0
x_init_obs = sliding_pack.db_sim.get_obs_slider_states(x_init_val, dtheta, ctact_config[0], ctact_config[1], beta)

# Solve the planning problem for nominal input
#  -------------------------------------------------------------------
import pdb; pdb.set_trace()
resultFlag, X_nom_val_opt, U_nom_val_opt, _, _, _ = optObjNom.solveProblem(
        0, [0., 0., 0.*(np.pi/180.), 0.], beta, [0.]*dyn.Nx,
        X_warmStart=X_nom_val)
if dyn.Nu > dynNom.Nu:
    U_nom_val_opt = cs.vertcat(
            U_nom_val_opt,
            cs.DM.zeros(np.abs(dyn.Nu - dynNom.Nu), N+N_MPC-1))
elif dynNom.Nu > dyn.Nu:
    U_nom_val_opt = U_nom_val_opt[:dyn.Nu, :]
f_d = cs.Function('f_d', [dyn.x, dyn.u], [dyn.x + dyn.f(dyn.x, dyn.u, beta)*dt])
f_rollout = f_d.mapaccum(N+N_MPC-1)
X_nom_comp = f_rollout([0., 0., 0., 0.], U_nom_val_opt)
#  ------------------------------------------------------------------

# define optimization problem
#  -------------------------------------------------------------------
optObj = sliding_pack.to.buildOptObj(
        dyn, N_MPC, tracking_config['TO'],
        X_nom_val, U_nom_val_opt, dt=dt,
)
#  -------------------------------------------------------------------

# Initialize variables for plotting
#  -------------------------------------------------------------------
X_plot = np.empty([dyn.Nx, Nidx])
U_plot = np.empty([dyn.Nu, Nidx-1])
del_plot = np.empty([dyn.Nz, Nidx-1])
X_plot[:, 0] = x_init_val
X_future = np.empty([dyn.Nx, N_MPC, Nidx])
comp_time = np.empty((Nidx-1, 1))
success = np.empty(Nidx-1)
cost_plot = np.empty((Nidx-1, 1))
# slider interaction
F_int_plot = np.empty([2, Nidx])
D_hat_plot = np.empty([dyn.Nx, Nidx])  # observed disturbance
D_true_plot = np.empty([dyn.Nx, Nidx])  # actual disturbance
#  -------------------------------------------------------------------
# Initialize variables for animation
#  -------------------------------------------------------------------
X_a_ani = np.empty([3, Nidx])
X_b_ani = np.empty([3, Nidx])
X_ctact_ani = np.empty([2, Nidx])
#  -------------------------------------------------------------------

#  Set selection matrix for X_goal
#  -------------------------------------------------------------------
if X_goal is None:
    S_goal_val = None
else:
    S_goal_idx = N_MPC-2
    S_goal_val = [0]*(N_MPC-1)
    S_goal_val[S_goal_idx] = 1
#  -------------------------------------------------------------------

# Set obstacles
#  ------------------------------------------------------------------
if optObj.numObs==0:
    obsCentre = None
    obsRadius = None
elif optObj.numObs==1:
    obsCentre = [[-0.27, 0.1]]
    # obsCentre = [[0., 0.28]]
    # obsCentre = [[0.2, 0.2]]
    obsRadius = [0.05]
#  ------------------------------------------------------------------

import pdb; pdb.set_trace()

# Set arguments and solve
#  -------------------------------------------------------------------
contactFlag = True
x0 = x_init_val
x0_obs = x_init_obs
# disturbance observer
N_exit = Nidx-2
d0 = [0.] * dyn.Nx
D_hat_plot[:, 0] = d0
D_true_plot[:, 0] = [0.]*dyn.Nx
for idx in range(Nidx-1):
    print('-------------------------')
    print(idx)
    # if idx == idxDist:
    #     print('i died here')
    #     x0[0] += 0.03
    #     x0[1] += -0.03
    #     x0[2] += 30.*(np.pi/180.)
    # ---- solve problem ----
    resultFlag, x_opt, u_opt, del_opt, f_opt, t_opt = optObj.solveProblem(
            idx, x0, beta, d0,
            S_goal_val=S_goal_val,
            obsCentre=obsCentre, obsRadius=obsRadius)
    print(f_opt)
    # ---- update initial state (simulation) ----
    u0 = u_opt[:, 0].elements()
    if contactFlag:
        # ---- build&solve the LCP problem ----
        # solve the contact force between two sliders
        u_f = double_dyn.solveLCP(np.r_[x0, x0_obs], u0, beta)
        # compute the augmented input
        u_all = np.insert(deepcopy(u0), 2, u_f)
        # update the states
        # true disturbance
        d_true = double_dyn.d_real(np.r_[x0, x0_obs], u_all, beta)
        x_all = double_dyn.updateStates(np.r_[x0, x0_obs], u_all, beta)
        x0 = deepcopy(x_all[0:4])
        x0_obs = deepcopy(x_all[4:8])
        # store interaction forces
        F_int_plot[:, idx] = u_f
        if x0_obs[3] >= 0.5*beta[1]:
            N_exit = idx
            import pdb; pdb.set_trace()
            contactFlag = False
    else:
        u_all = np.insert(deepcopy(u0), 2, [0., 0.])
        d_true = double_dyn.d_real(np.r_[x0, x0_obs], u_all, beta)
        x0 = (x0 + dyn.f(x0, u0, beta)*dt).elements()
        F_int_plot[:, idx] = [0., 0.]
    # update estimated disturbance
    # d0 = (np.array(d0)+K_d*(x0 - x_opt[:, 1] - np.array(d0)*dt) / dt).elements()
    d0 = (np.array(d0)+K_d*(x0 - x_opt[:, 1]) / dt).elements()
    D_hat_plot[:, idx+1] = d0
    D_true_plot[:, idx+1] = d_true.elements()
    # ---- store values for plotting ----
    comp_time[idx] = t_opt
    success[idx] = resultFlag
    cost_plot[idx] = f_opt
    X_plot[:, idx+1] = x0
    U_plot[:, idx] = u0
    X_future[:, :, idx] = np.array(x_opt)
    # ---- store values for animation ----
    X_a_ani[:, idx] = x0[0:3]
    X_b_ani[:, idx] = x0_obs[0:3]
    X_ctact_ani[:, idx] = [-0.5*beta[0], -0.5*beta[0]*np.tan(x0[3])]
    if dyn.Nz > 0:
        del_plot[:, idx] = del_opt[:, 0].elements()
    # ---- update selection matrix ----
    if X_goal is not None and f_opt < 0.00001 and S_goal_idx > 10:
        S_goal_idx -= 1
        S_goal_val = [0]*(N_MPC-1)
        S_goal_val[S_goal_idx] = 1
        print(S_goal_val)
        # sys.exit()
#  -------------------------------------------------------------------

import pdb; pdb.set_trace()

fig, axs = plt.subplots(2, 4, sharex=True)
fig.set_size_inches(10*(2/3), 10, forward=True)
t_Nx = np.linspace(0, T, N)
t_Nu = np.linspace(0, T, N-1)

# plot interaction force
for i in range(2):
    axs[0, i].plot(t_Nu, F_int_plot[i, 0:N-1].T, color='red',
                   linestyle='--', label='lcp')
    handles, labels = axs[0, i].get_legend_handles_labels()
    axs[0, i].legend(handles, labels)
    axs[0, i].set_xlabel('time [s]')
    axs[0, i].set_ylabel('f_n' if i == 0 else 'f_t')
    axs[0, i].grid()
    
for i in range(dyn.Nx):
    axs[1, i].plot(t_Nx, D_hat_plot[i, 0:N].T, color='red',
                   linestyle='--', label='pred')
    axs[1, i].plot(t_Nx, D_true_plot[i, 0:N].T, color='blue',
                   linestyle='--', label='real')
    handles, labels = axs[1, i].get_legend_handles_labels()
    axs[1, i].legend(handles, labels)
    axs[1, i].set_xlabel('time [s]')
    axs[1, i].set_ylabel('d_hat_{}'.format(i))
    axs[1, i].grid()
    
plt.show()

simObj = sliding_pack.db_sim.buildDoubleSliderSimObj(
        N_exit, double_slider_config['dynamics'], 0.04, method='qpoases',
        showAnimFlag=True, saveFileFlag=True)

simObj.visualization(X_a_ani.T, X_b_ani.T, X_ctact_ani.T, beta)

# exit(0)

# show sparsity pattern
# sliding_pack.plots.plot_sparsity(cs.vertcat(*opt.g), cs.vertcat(*opt.x), xu_opt)
p_new = cs.Function('p_new', [dyn.x], [dyn.p(dyn.x, beta)])
p_map = p_new.map(N)
X_pusher_opt = p_map(X_plot)
#  -------------------------------------------------------------------

if save_to_file:
    #  Save data to file using pandas
    #  -------------------------------------------------------------------
    df_state = pd.DataFrame(
                    np.concatenate((
                        np.array(X_nom_val[:, :Nidx]).transpose(),
                        np.array(X_plot).transpose(),
                        np.array(X_pusher_opt).transpose()
                        ), axis=1),
                    columns=['x_nom', 'y_nom', 'theta_nom', 'psi_nom',
                             'x_opt', 'y_opt', 'theta_opt', 'psi_opt',
                             'x_pusher', 'y_pusher'])
    df_state.index.name = 'idx'
    df_state.to_csv('tracking_circle_cc_state.csv',
                    float_format='%.5f')
    time = np.linspace(0., T, Nidx-1)
    print('********************')
    print(U_plot.transpose().shape)
    print(cost_plot.shape)
    print(comp_time.shape)
    print(time.shape)
    print(time[:, None].shape)
    df_action = pd.DataFrame(
                    np.concatenate((
                        U_plot.transpose(),
                        time[:, None],
                        cost_plot,
                        comp_time
                        ), axis=1),
                    columns=['u0', 'u1', 'u3', 'u4',
                    # columns=['u0', 'u1', 'u3',
                             'time', 'cost', 'comp_time'])
    df_action.index.name = 'idx'
    df_action.to_csv('tracking_circle_cc_action.csv',
                     float_format='%.5f')
    #  -------------------------------------------------------------------

# Animation
#  -------------------------------------------------------------------
plt.rcParams['figure.dpi'] = 150
if show_anim:
    #  ---------------------------------------------------------------
    fig, ax = sliding_pack.plots.plot_nominal_traj(
                x0_nom[:Nidx], x1_nom[:Nidx], plot_title='')
    # add computed nominal trajectory
    X_nom_val_opt = np.array(X_nom_val_opt)
    # ax.plot(X_nom_val_opt[0, :], X_nom_val_opt[1, :], color='blue',
    #         linewidth=2.0, linestyle='dashed')
    X_nom_comp = np.array(X_nom_comp)
    # ax.plot(X_nom_comp[0, :], X_nom_comp[1, :], color='green',
    #         linewidth=2.0, linestyle='dashed')
    # add obstacles
    if optObj.numObs > 0:
        for i in range(len(obsCentre)):
            circle_i = plt.Circle(obsCentre[i], obsRadius[i], color='b')
            ax.add_patch(circle_i)
    # set window size
    fig.set_size_inches(8, 6, forward=True)
    # get slider and pusher patches
    dyn.set_patches(ax, X_plot, beta)
    # call the animation
    ani = animation.FuncAnimation(
            fig,
            dyn.animate,
            fargs=(ax, X_plot, beta, False, None, X_future),
            frames=Nidx-1,
            interval=dt*1000,  # microseconds
            blit=True,
            repeat=False,
    )
    # to save animation, uncomment the line below:
    ani.save('./video/MPC_MPCC_line.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(3, 4, sharex=True)
fig.set_size_inches(10, 10, forward=True)
t_Nx = np.linspace(0, T, N)
t_Nu = np.linspace(0, T, N-1)
t_idx_x = t_Nx[0:Nidx]
t_idx_u = t_Nx[0:Nidx-1]
ctrl_g_idx = dyn.g_u.map(Nidx-1)
ctrl_g_val = ctrl_g_idx(U_plot, del_plot)
#  -------------------------------------------------------------------
# plot position
for i in range(dyn.Nx):
    axs[0, i].plot(t_Nx, X_nom_val[i, 0:N].T, color='red',
                   linestyle='--', label='nom')
    axs[0, i].plot(t_Nx, X_nom_val_opt[i, 0:N].T, color='blue',
                   linestyle='--', label='plan')
    axs[0, i].plot(t_idx_x, X_plot[i, :], color='orange', label='mpc')
    handles, labels = axs[0, i].get_legend_handles_labels()
    axs[0, i].legend(handles, labels)
    axs[0, i].set_xlabel('time [s]')
    axs[0, i].set_ylabel('x%d' % i)
    axs[0, i].grid()
#  -------------------------------------------------------------------
# plot computation time
axs[1, 0].plot(t_idx_u, comp_time, color='b')
handles, labels = axs[1, 0].get_legend_handles_labels()
axs[1, 0].legend(handles, labels)
axs[1, 0].set_xlabel('time [s]')
axs[1, 0].set_ylabel('comp time [s]')
axs[1, 0].grid()
#  -------------------------------------------------------------------
# plot computation cost
axs[1, 1].plot(t_idx_u, cost_plot, color='b', label='cost')
handles, labels = axs[1, 1].get_legend_handles_labels()
axs[1, 1].legend(handles, labels)
axs[1, 1].set_xlabel('time [s]')
axs[1, 1].set_ylabel('cost')
axs[1, 1].grid()
#  -------------------------------------------------------------------
# plot extra variables
for i in range(dyn.Nz):
    axs[1, 2].plot(t_idx_u, del_plot[i, :].T, label='s%d' % i)
handles, labels = axs[1, 2].get_legend_handles_labels()
axs[1, 2].legend(handles, labels)
axs[1, 2].set_xlabel('time [s]')
axs[1, 2].set_ylabel('extra vars')
axs[1, 2].grid()
#  -------------------------------------------------------------------
# plot constraints
for i in range(dyn.Ng_u):
    axs[1, 3].plot(t_idx_u, ctrl_g_val[i, :].T, label='g%d' % i)
handles, labels = axs[1, 3].get_legend_handles_labels()
axs[1, 3].legend(handles, labels)
axs[1, 3].set_xlabel('time [s]')
axs[1, 3].set_ylabel('constraints')
axs[1, 3].grid()
#  -------------------------------------------------------------------
# plot actions
for i in range(dyn.Nu):
    axs[2, i].plot(t_Nu, U_nom_val_opt[i, 0:N-1].T, color='blue',
                   linestyle='--', label='plan')
    axs[2, i].plot(t_idx_u, U_plot[i, :], color='orange', label='mpc')
    handles, labels = axs[2, i].get_legend_handles_labels()
    axs[2, i].legend(handles, labels)
    axs[2, i].set_xlabel('time [s]')
    axs[2, i].set_ylabel('u%d' % i)
    axs[2, i].grid()
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
import pdb; pdb.set_trace()
plt.show()
#  -------------------------------------------------------------------