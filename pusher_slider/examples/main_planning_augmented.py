# Author: Joao Moura (Modifed by Yongpeng Jiang)
# Date: 21/08/2020 (Modified on 16/12/2022)
#  -------------------------------------------------------------------
# Description:
#  This script implements an augmented version of non-linear program (NLP)
#  model predictive controller (MPC) for tracking a trajectory of multiple
#  square slider objects with a single and sliding contact pusher.
#  -------------------------------------------------------------------

#  import libraries
#  -------------------------------------------------------------------
import sys
import yaml
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
planning_config = sliding_pack.load_config('planning_config_augmented.yaml')
#  -------------------------------------------------------------------

# Set Problem constants
#  -------------------------------------------------------------------
T = 2.5  # time of the simulation is seconds
freq = 25  # number of increments per second
contactFace = '-x'
pusherAngleLim = 0.
sliderRelAngleLim = 0.
show_anim = True
save_to_file = False
#  -------------------------------------------------------------------
# Computing Problem constants
#  -------------------------------------------------------------------
dt = 1.0/freq  # sampling time
N = int(T*freq)  # total number of iterations
#  -------------------------------------------------------------------
beta = [
    planning_config['dynamics']['xLenght'],
    planning_config['dynamics']['yLenght'],
    planning_config['dynamics']['pusherRadious']
]

# define system dynamics
#  -------------------------------------------------------------------
dyn = sliding_pack.dyn.Aug_Sys_sq_slider_quasi_static_ellip_lim_surf(
        planning_config['dynamics'],
        planning_config['TO']['contactMode'],
        contactFace,
        pusherAngleLim,
        sliderRelAngleLim,
        beta
)

#  -------------------------------------------------------------------

# Generate Nominal Trajectory
#  -------------------------------------------------------------------
if planning_config['TO']['X_goal']:
    X_goal = planning_config['TO']['X_goal']
else:
    X_goal = [0.4, 0.2, (20./180.) * np.pi, 0.]
    # X_goal = [0.4, 0.1, 0.3, 0.]
X_goal = sliding_pack.aug.augment_state(X_goal, beta, sliderRelAngleLim)
print('X_goal: ', X_goal)

x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(X_goal[0], X_goal[1], N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.3, 0.4, N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.1, N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_eight(0.2, N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_sine(0.0, 1.0, 0.5, 0.05, N, 0)
#  -------------------------------------------------------------------

# stack state and derivative of state
# the slider rotate to tangent direction of the nominal trajectory in one step
X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt, beta,
                                                   sliderRelAngleLim,
                                                   multiSliders=True)
#  ------------------------------------------------------------------

# Compute nominal actions for sticking contact
#  ------------------------------------------------------------------
optObj = sliding_pack.to.buildOptObj(
        dyn, N, planning_config['TO'], X_nom_val=X_nom_val, dt=dt,
        useGoalFlag=False, phic0Fixed=True, multiSliders=True)
# Set obstacles
#  ------------------------------------------------------------------
if optObj.numObs==0:
    obsCentre = None
    obsRadius = None
elif optObj.numObs==2:
    obsCentre = [[0.2, 0.2], [0.1, 0.5]]
    obsRadius = [0.05, 0.05]
elif optObj.numObs==3:
    obsCentre = [[0.1, 0.1], [0.0, 0.3], [0.3, -0.125]]
    obsRadius = [0.05, 0.05, 0.05]
#  ------------------------------------------------------------------
x_init = [0., 0., 0., 0.8]
# x_init = [0., 0., -20.*(np.pi/180.), -50.*(np.pi/180.)]
# x_init = [0.38, 0.22, -70.*(np.pi/180.), 0.]
# x_init = [0., 0., 340.*(np.pi/180.), 0.]
# X_goal = X_nom_val[:, -1].toarray().squeeze().tolist()
x_init = sliding_pack.aug.augment_state(x_init, beta, sliderRelAngleLim)
import pdb; pdb.set_trace()
resultFlag, X_nom_val_opt, U_nom_val_opt, other_opt, _, t_opt = optObj.solveProblem(
        0, x_init, beta,
        X_warmStart=X_nom_val,
        obsCentre=obsCentre, obsRadius=obsRadius,
        X_goal_val=X_goal)

import pdb; pdb.set_trace()

f_d = cs.Function('f_d', [dyn.x, dyn.u],
                         [cs.vertcat(dyn.x[:4] + dyn.f1(dyn.x, dyn.u, beta)*dt, 
                          dyn.x[:3] + dyn.f1(dyn.x, dyn.u, beta)[:3]*dt + beta[0]*cs.vertcat(1., cs.tan(dyn.x[2]), 0.),
                          dyn.x[7])])
f_rollout = f_d.mapaccum(N-1)
print('comp time: ', t_opt)
p_new = cs.Function('p_new', [dyn.x], [dyn.p(dyn.x, beta)])
p_map = p_new.map(N)
X_pusher_opt = p_map(X_nom_val_opt)
#  ------------------------------------------------------------------


if save_to_file:
    #  Save data to file using pandas
    #  -------------------------------------------------------------------
    df_state = pd.DataFrame(
                    np.array(cs.vertcat(X_nom_val_opt,X_pusher_opt)).transpose(),
                    columns=['x_slider', 'y_slider', 'theta_slider', 'psi_pusher', 'x_pusher', 'y_pusher'])
    df_state.to_csv('planning_positive_angle_state.csv',
                    float_format='%.5f')
    if planning_config['TO']['contactMode'] == 'sticking':
        column_labels = ['u0', 'u1']
    else:
        column_labels = ['u0', 'u1', 'u2', 'u3']
        
    df_action = pd.DataFrame(
                    np.array(U_nom_val_opt).transpose(),
                    columns=column_labels)
    df_action.to_csv('planning_positive_angle_action.csv',
                     float_format='%.5f')
    #  -------------------------------------------------------------------

import pdb; pdb.set_trace()

# Animation
#  -------------------------------------------------------------------
plt.rcParams['figure.dpi'] = 150
if show_anim:
    #  ---------------------------------------------------------------
    fig, ax = sliding_pack.plots.plot_nominal_traj(
                x0_nom, x1_nom, plot_title='')
    # add computed nominal trajectory
    X_nom_val_opt = np.array(X_nom_val_opt)
    ax.plot(X_nom_val_opt[0, :], X_nom_val_opt[1, :], color='blue',
            linewidth=2.0, linestyle='dashed')
    # add obstacles
    if optObj.numObs > 0:
        for i in range(len(obsCentre)):
            circle_i = plt.Circle(obsCentre[i], obsRadius[i], color='b')
            ax.add_patch(circle_i)
    # set window size
    fig.set_size_inches(8, 6, forward=True)
    # get slider and pusher patches
    dyn.set_patches(ax, X_nom_val_opt, beta)
    # call the animation
    ani = animation.FuncAnimation(
            fig,
            dyn.animate,
            fargs=(ax, X_nom_val_opt, beta),
            frames=N-1,
            interval=dt*1000*1,  # microseconds
            blit=True,
            repeat=False,
    )
    # to save animation, uncomment the line below:
    ani.save('planning_double_slider.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

# # Plot Optimization Results
# #  -------------------------------------------------------------------
t_Nx = np.linspace(0, T, N)
t_Nu = np.linspace(0, T, N-1)
ctrl_g = dyn.g_u.map(N-1)
ctrl_g_val = ctrl_g(X_nom_val_opt[:, :-1], U_nom_val_opt, other_opt, beta)

# # Figure1: plot state and slack variables
fig1, axs1 = plt.subplots(2, 4, sharex=True)
fig1.set_size_inches(10, 10, forward=True)
# #  -------------------------------------------------------------------
# # plot position (slider1)
for i in range(4):
    axs1[0, i].plot(t_Nx, X_nom_val[i, 0:N].T, color='red',
                   linestyle='--', label='nom')
    axs1[0, i].plot(t_Nx, X_nom_val_opt[i, 0:N].T, color='blue',
                   linestyle='--', label='plan')
    handles, labels = axs1[0, i].get_legend_handles_labels()
    axs1[0, i].legend(handles, labels)
    axs1[0, i].set_xlabel('time [s]')
    axs1[0, i].set_ylabel('x%d' % i)
    axs1[0, i].grid('on')
# #  -------------------------------------------------------------------
# # plot extra variables
for i in range(dyn.Nz):
    axs1[1, 2].plot(t_Nu, other_opt[i, :].T, label='s%d' % i)
handles, labels = axs1[1, 2].get_legend_handles_labels()
axs1[1, 2].legend(handles, labels)
axs1[1, 2].set_xlabel('time [s]')
axs1[1, 2].set_ylabel('extra vars')
axs1[1, 2].grid('on')
# #  -------------------------------------------------------------------

# # Figure2: plot input
fig2, axs2 = plt.subplots(2, 4, sharex=True)
fig2.set_size_inches(10, 10, forward=True)
# #  -------------------------------------------------------------------
# # plot actions
for i in range(dyn.Nu):
    axs2[i // 4, i % 4].plot(t_Nu, U_nom_val_opt[i, 0:N-1].T, color='blue',
                   linestyle='--', label='plan')
    handles, labels = axs2[i // 4, i % 4].get_legend_handles_labels()
    axs2[i // 4, i % 4].legend(handles, labels)
    axs2[i // 4, i % 4].set_xlabel('time [s]')
    axs2[i // 4, i % 4].set_ylabel('u%d' % i)
    axs2[i // 4, i % 4].grid('on')
# #  -------------------------------------------------------------------

fig3, axs3 = plt.subplots(2, 4, sharex=True)
fig3.set_size_inches(10, 10, forward=True)
# #  -------------------------------------------------------------------
# # plot constraints (friction cone)
idx_Ng_u_cone = [(0, 0), (0, 1), (1, 4), (1, 5), (2, 6), (2, 7)]  # (figure column index, Ng_u_index)
for col, i in idx_Ng_u_cone:
    axs3[0, col].plot(t_Nu, ctrl_g_val[i, :].T, label='g%d' % i)
    handles, labels = axs3[0, col].get_legend_handles_labels()
    axs3[0, col].legend(handles, labels)
    axs3[0, col].set_xlabel('time [s]')
    axs3[0, col].set_ylabel('cst.cone')
    axs3[0, col].grid('on')
# #  -------------------------------------------------------------------
# # plot constraints (complementary)
idx_Ng_u_cone = [(3, 2), (3, 3)]  # (figure column index, Ng_u_index)
for col, i in idx_Ng_u_cone:
    axs3[0, col].plot(t_Nu, ctrl_g_val[i, :].T, label='g%d' % i)
    handles, labels = axs3[0, col].get_legend_handles_labels()
    axs3[0, col].legend(handles, labels)
    axs3[0, col].set_xlabel('time [s]')
    axs3[0, col].set_ylabel('cst.cc')
    axs3[0, col].grid('on')
# #  -------------------------------------------------------------------
# # plot constraints (balance)
idx_Ng_u_cone = [(0, 8), (0, 9), (1, 10)]  # (figure column index, Ng_u_index)
for col, i in idx_Ng_u_cone:
    axs3[1, col].plot(t_Nu, ctrl_g_val[i, :].T, label='g%d' % i)
    handles, labels = axs3[1, col].get_legend_handles_labels()
    axs3[1, col].legend(handles, labels)
    axs3[1, col].set_xlabel('time [s]')
    axs3[1, col].set_ylabel('cst.balance')
    axs3[1, col].grid('on')
# #  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------
