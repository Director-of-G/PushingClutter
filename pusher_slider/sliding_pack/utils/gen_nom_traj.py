## Author: Joao Moura (Modifed by Yongpeng Jiang)
## Contact: jpousad@ed.ac.uk (jyp19@mails.tsinghua.edu.cn)
## Date: 15/12/2020 (Modified on 16/12/2022)
## -------------------------------------------------------------------
## Description:
## 
## Functions for outputting different nominal trajectories
## -------------------------------------------------------------------

## -------------------------------------------------------------------
## Import libraries
## -------------------------------------------------------------------
import numpy as np
import casadi as cs
from copy import deepcopy
from rrt_pack.utilities.geometry import rotation_matrix

## Generate Nominal Trajectory (line)
def generate_traj_line(x_f, y_f, N, N_MPC):
    x_nom = np.linspace(0.0, x_f, N)
    y_nom = np.linspace(0.0, y_f, N)
    # return x_nom, y_nom
    return np.concatenate((x_nom, x_f+x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_f+y_nom[1:N_MPC+1]), axis=0)
def generate_traj_circle(theta_i, theta_f, radious, N, N_MPC):
    s = np.linspace(theta_i, theta_f, N)
    x_nom = radious*np.cos(s)
    y_nom = radious*np.sin(s)
    # initial position at the origin
    x_nom -= x_nom[0]
    y_nom -= y_nom[0]
    # return x_nom, y_nom
    return np.concatenate((x_nom, x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_nom[1:N_MPC+1]), axis=0)
def generate_dubins_curve(theta_i, dtheta, center_x, center_y, radious, N, N_MPC):
    s = np.linspace(theta_i, theta_i + dtheta, N)
    x_nom = center_x+radious*np.cos(s)
    y_nom = center_y+radious*np.sin(s)
    # initial position at the origin
    x_nom -= x_nom[0]
    y_nom -= y_nom[0]
    # return x_nom, y_nom
    return np.concatenate((x_nom, x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_nom[1:N_MPC+1]), axis=0)
def generate_traj_ellipse(theta_i, theta_f, radious_x, radious_y, N, N_MPC):
    s = np.linspace(theta_i, theta_f, N)
    x_nom = radious_x*np.cos(s)
    y_nom = radious_y*np.sin(s)
    # initial position at the origin
    x_nom -= x_nom[0]
    y_nom -= y_nom[0]
    # return x_nom, y_nom
    return np.concatenate((x_nom, x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_nom[1:N_MPC+1]), axis=0)
def generate_traj_eight(side_lenght, N, N_MPC):
    s = np.linspace(0.0, 2*np.pi, N)
    x_nom = side_lenght*np.sin(s)
    y_nom = side_lenght*np.sin(s)*np.cos(s)
    # return x_nom, y_nom
    return np.concatenate((x_nom, x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_nom[1:N_MPC+1]), axis=0)
def generate_traj_sine(yaw, lamda, length, amplitude, N, N_MPC):
    s = np.linspace(0.0, 2.0 * np.pi, N)  # 2 periods
    x_nom = s / np.max(s) * length
    y_nom = amplitude * np.sin(s / lamda)
    x_nom -= x_nom[0]
    y_nom -= y_nom[0]
    coords = np.concatenate((np.expand_dims(x_nom, axis=0), np.expand_dims(y_nom, axis=0)), axis=0)
    coords = rotation_matrix(yaw) @ coords
    x_nom = coords[0].squeeze()
    y_nom = coords[1].squeeze()
    return np.concatenate((x_nom, x_nom[-1]+x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_nom[-1]+y_nom[1:N_MPC+1]), axis=0)
def compute_nomState_from_nomTraj(x_data, y_data, dt, beta=None, phi_r=0., multiSliders=False):
    # assign two first state trajectories
    x0_nom = x_data
    x1_nom = y_data
    # compute diff for planar traj
    Dx0_nom = np.diff(x0_nom)
    Dx1_nom = np.diff(x1_nom)
    # compute traj angle 
    ND = len(Dx0_nom)
    x2_nom = np.empty(ND)
    theta = 0.0
    for i in range(ND):
        # align +x axis with the forwarding direction
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, s), (-s, c)))
        Dx_new = R.dot(np.array((Dx0_nom[i],Dx1_nom[i])))
        print(Dx_new)
        theta += np.arctan2(Dx_new[1], Dx_new[0])
        x2_nom[i] = theta
    x2_nom = np.append(x2_nom, x2_nom[-1])
    Dx2_nom = np.diff(x2_nom)
    # specify angle of the pusher relative to slider
    x3_nom = np.zeros(x0_nom.shape)
    Dx3_nom = np.diff(x3_nom)
    # stack state and derivative of state
    # x_nom = np.vstack((x0_nom, x1_nom, x2_nom, x3_nom))
    if not multiSliders:
        x_nom = cs.horzcat(x0_nom, x1_nom, x2_nom, x3_nom).T
        dx_nom = cs.horzcat(Dx0_nom, Dx1_nom, Dx2_nom, Dx3_nom).T/dt
    else:
        x4_nom = x0_nom + beta[0]
        x5_nom = x1_nom + beta[0]*np.tan(phi_r)
        x6_nom = deepcopy(x2_nom)
        x7_nom = np.ones_like(x3_nom)*phi_r
        Dx4_nom, Dx5_nom, Dx6_nom, Dx7_nom = np.diff(x4_nom), np.diff(x5_nom), np.diff(x6_nom), np.diff(x7_nom)
        x_nom = cs.horzcat(x0_nom, x1_nom, x2_nom, x3_nom, x4_nom, x5_nom, x6_nom, x7_nom).T
        dx_nom = cs.horzcat(Dx0_nom, Dx1_nom, Dx2_nom, Dx3_nom, Dx4_nom, Dx5_nom, Dx6_nom, Dx7_nom).T/dt
    return x_nom, dx_nom
