# Author: Yongpeng Jiang
# Date: 12/15/2022
#  -------------------------------------------------------------------
# Description:
#  This script implements a self-supervised data collection process
#  for the pusher-slider system. The pusher exerts force respectively
#  on each face of the slider, and the full system states are recorded.
#  -------------------------------------------------------------------
#  import libraries
#  -------------------------------------------------------------------
import casadi as cs
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import tqdm
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

# Get config files
#  -------------------------------------------------------------------
data_config = sliding_pack.load_config('data_config.yaml')
tracking_config = sliding_pack.load_config('tracking_config.yaml')
planning_config = sliding_pack.load_config('planning_config.yaml')
#  -------------------------------------------------------------------

class DataCollector(object):
    def __init__(self, config, beta) -> None:
        self.radius = config['regionRadius']
        self.yaw_resol = config['yawResolution']
        self.theta_resol = config['thetaResolution']
        self.push_pt_resol = config['pushPointResolution']
        self.max_iters = config['maxIters']
        self.contact_face = config['contactFace']
        self.N = int(200)  # time horizon
        self.N_MPC = int(15)  # MPC time window
        self.dt = 0.01  # time step

        self.beta = beta

    def make_samples(self):
        """
        Take samples on the circle, considering slider's rotation,
        w.r.t. the given resolution.
        """
        yaw = np.arange(-0.5*np.pi, 0.5*np.pi, step=self.yaw_resol)
        goal_xy = self.radius * np.c_[np.cos(yaw), np.sin(yaw)]
        goal_theta = np.arange(-0.5*np.pi, 0.5*np.pi + self.theta_resol, step=self.theta_resol)
        pusher_loc = {'x': np.arange(-0.5 * self.beta[1], 0.5 * self.beta[1] + self.push_pt_resol, step=self.push_pt_resol),
                      'y': np.arange(-0.5 * self.beta[0], 0.5 * self.beta[0], step=self.push_pt_resol)}

        return goal_xy, goal_theta, pusher_loc

    def collect_data(self):
        """
        Collect data in a self-supervised manner.
        """
        data = np.zeros((0, 10, 3))
        # take samples
        goal_xy, goal_theta, pusher_loc = self.make_samples()

        plt.ion()

        # traverse all samples
        for face in self.contact_face:
            pusher_loc_face = pusher_loc[face[-1]]  # face[-1] = 'x' or 'y'
            for xy in goal_xy:
                # for theta in goal_theta:
                theta = np.arctan2(xy[1], xy[0])
                for loc in pusher_loc_face:
                    psi = -np.arctan2(loc, self.beta[0]/2)
                    if psi >= 0.9 or psi <= -0.9:
                        continue
                    x_init_val = [0., 0., 0., psi]
                    X_goal = [xy[0], xy[1], theta, 0.]
                    # define system dynamics
                    dynTrack = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
                        tracking_config['dynamics'],
                        tracking_config['TO']['contactMode'],
                        '-x',
                        pusherAngleLim=psi
                    )
                    dynPlan = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
                        planning_config['dynamics'],
                        planning_config['TO']['contactMode'],
                        '-x',
                        pusherAngleLim=psi
                    )
                    # compute nominal states
                    x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(xy[0], xy[1], self.N, self.N_MPC)
                    X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, self.dt)
                    X_nom_val[2, :] = np.linspace(x_init_val[2], X_goal[2], X_nom_val.shape[1])
                    # build planning problem
                    optObjPlan = sliding_pack.to.buildOptObj(
                        dynPlan, self.N, planning_config['TO'], dt=self.dt, X_nom_val=X_nom_val,
                        maxIter=self.max_iters, useGoalFlag=False, phic0Fixed=True
                    )
                    # solve the planning problem
                    resultFlag, X_nom_val_opt, U_nom_val_opt, _, _, _ = optObjPlan.solveProblem(
                        0, x_init_val, self.beta, X_warmStart=X_nom_val, X_goal_val=X_goal
                    )

                    data = np.concatenate((data, np.expand_dims(X_nom_val_opt.toarray()[:3, ::20].T, axis=0)), axis=0)
                    """
                    # get nominal variables for tracking
                    if dynTrack.Nu > dynPlan.Nu:
                        U_nom_val_opt = cs.vertcat(
                                U_nom_val_opt,
                                cs.DM.zeros(np.abs(dynTrack.Nu - dynPlan.Nu), self.N+self.N_MPC-1))
                    elif dynPlan.Nu > dynTrack.Nu:
                        U_nom_val_opt = U_nom_val_opt[:dynTrack.Nu, :]
                    # build tracking problem
                    optObjTrack = sliding_pack.to.buildOptObj(
                        dynTrack, self.N_MPC, tracking_config['TO'],
                        X_nom_val_opt, U_nom_val_opt, dt=self.dt,
                        useGoalFlag=False, maxIter=self.max_iters
                    )
                    solved_X = np.empty([dynTrack.Nx, self.N])
                    solved_U = np.empty([dynTrack.Nu, self.N])
                    solved_X[:, 0] = x_init_val
                    # set selection matrix for goal
                    S_goal_idx = self.N_MPC-2
                    S_goal_val = [0]*(self.N_MPC-1)
                    S_goal_val[S_goal_idx] = 1
                    x0 = x_init_val
                    # solve the tracking problem
                    for idx in range(self.N-1):
                        resultFlag, x_opt, u_opt, del_opt, f_opt, t_opt = optObjTrack.solveProblem(
                            idx, x0=x0, beta=self.beta,
                            obsCentre=None, obsRadius=None,
                            S_goal_val=None, X_goal_val=X_goal
                        )
                        u0 = u_opt[:, 0].elements()
                        x0 = (x0 + dynTrack.f(x0, u0, beta)*self.dt).elements()
                        solved_X[:, idx+1] = x0
                        solved_U[:, idx] = u0
                    # get the pusher contact points
                    p_new = cs.Function('p_new', [dynTrack.x], [dynTrack.p(dynTrack.x, self.beta)])
                    p_map = p_new.map(self.N)
                    solved_pusher = p_map(solved_X)
                    """

                    # show animation
                    #  ---------------------------------------------------------------
                    fig, ax = sliding_pack.plots.plot_nominal_traj(
                                x0_nom, x1_nom, plot_title='')
                    # add computed nominal trajectory
                    X_nom_val_opt = np.array(X_nom_val_opt)
                    ax.plot(X_nom_val_opt[0, :], X_nom_val_opt[1, :], color='blue',
                            linewidth=2.0, linestyle='dashed')
                    X_nom_val = np.array(X_nom_val)
                    ax.plot(X_nom_val[0, :], X_nom_val[1, :], color='orange',
                            linewidth=2.0, linestyle='dashed')
                    # set window size
                    fig.set_size_inches(8, 6, forward=True)
                    # get slider and pusher patches
                    dynTrack.set_patches(ax, X_nom_val_opt, self.beta)
                    # call the animation
                    ani = animation.FuncAnimation(
                            fig,
                            dynTrack.animate,
                            fargs=(ax, X_nom_val_opt, self.beta),
                            frames=self.N-1,
                            interval=self.dt*1,  # microseconds
                            blit=True,
                            repeat=False,
                    )
                    
                    plt.pause(6)
                    plt.close()

                temp_data = data.reshape(-1, 3)
                plt.scatter(temp_data[:, 0], temp_data[:, 1])
                plt.gca().set_aspect('equal')
                import pdb; pdb.set_trace()


if __name__ == '__main__':
    beta = [
        planning_config['dynamics']['xLenght'],
        planning_config['dynamics']['yLenght'],
        planning_config['dynamics']['pusherRadious']
    ]
    collector = DataCollector(config=data_config['collection'],
                              beta=beta)
    collector.collect_data()
