# Author: Yongpeng Jiang
# Date: 12/10/2022
#  -------------------------------------------------------------------
# Description:
#  Do numerical rollout, simulate the slider's dynamics and test the 
#  differential flatness properties.
#  -------------------------------------------------------------------

import casadi as cs
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------
from rrt_pack.search_space.planar_search_space import PlanarSearchSpace
#  -------------------------------------------------------------------

planning_config = sliding_pack.load_config('planning_switch_config.yaml')

# Set Problem constants
#  -------------------------------------------------------------------
T = 0.2  # time of the simulation is seconds
freq = 1000  # number of increments per second
show_anim = True
save_to_file = True
#  -------------------------------------------------------------------
# Computing Problem constants
#  -------------------------------------------------------------------
dt = 1.0/freq  # sampling time
N = int(T*freq)  # total number of iterations
#  -------------------------------------------------------------------

# define desired inputs
#  -------------------------------------------------------------------
U_nom = {'force': [0.1, 0.018],
         'psi': -0.08}

# define system dynamics
#  -------------------------------------------------------------------
dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
    planning_config['dynamics'],
    'sticking',
    '-x',
    pusherAngleLim=0.
)

# define rollout symbolic functions
#  -------------------------------------------------------------------
beta = [
    planning_config['dynamics']['xLenght'],
    planning_config['dynamics']['yLenght'],
    planning_config['dynamics']['pusherRadious']
]

X_dimensions = np.array([(0, 0.5), (0, 0.5), (-np.pi, np.pi)])
Obstacles = np.array([(0.05, 0.0, 0.5, 0.05), 
                        (0.0, 0.0, 0.05, 0.5),
                        (0.05, 0.45, 0.5, 0.5),
                        (0.45, 0.05, 0.5, 0.45),
                        (0.15, 0.15, 0.45, 0.35)])
X = PlanarSearchSpace(X_dimensions, Obstacles)
X.create_slider_geometry(geom=[0.07, 0.12])
X.create_slider_dynamics(ratio = 1 / 726.136, miu=0.2)

f_d = cs.Function('f_d', [dyn.x, dyn.u], [dyn.x + dyn.f(dyn.x, dyn.u, beta)*dt])
f_rollout = f_d.mapaccum(N-1)
p_new = cs.Function('p_new', [dyn.x], [dyn.p(dyn.x, beta)])
p_map = p_new.map(N)

U = np.array(U_nom['force']) / np.linalg.norm(U_nom['force'], ord=2)
# U = np.expand_dims(np.append(force, 0.), axis=1)
U = np.expand_dims(U, axis=1)
U = U.repeat(N-1, 1)

X0 = np.array([0., 0., 0., U_nom['psi']])

X_rollout = f_rollout(X0, U).toarray()

# calculate revolution using differential flatness
revol = X.pose2steer(start=X_rollout[:3, 0], end=X_rollout[:3, -1])
print('revol=(x: {0}, y:{1}, theta: {2})'.format(revol.x, revol.y, revol.theta))

# animation
#  ---------------------------------------------------------------
# set window size
fig, ax = plt.subplots()
fig.set_size_inches(8, 6, forward=True)
# get slider and pusher patches
dyn.set_patches(ax, X_rollout, beta, True, U)
# call the animation
ani = animation.FuncAnimation(
    fig,
    dyn.animate,
    fargs=(ax, X_rollout, beta, True, U),
    frames=N-1,
    interval=dt*1000,  # microseconds
    blit=True,
    repeat=False,
)

margin_width=0.1
x_data = X_rollout[0, :]
y_data = X_rollout[1, :]
ax.set_xlim((np.min(x_data) - margin_width, np.max(x_data) + margin_width))
ax.set_ylim((np.min(y_data) - margin_width, np.max(y_data) + margin_width))
ax.set_autoscale_on(False)
ax.grid()
ax.set_aspect('equal', 'box')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')

ani.save('./video/test_kinematics.mp4', fps=25)

plt.show()
