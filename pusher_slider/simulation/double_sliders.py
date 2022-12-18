# Author: Yongpeng Jiang
# Date: 17/11/2022
#  -------------------------------------------------------------------
# Description:
#  This script implements a simulation base on limit surface and LCP
#  modeling for simulating double sliders.
#  -------------------------------------------------------------------

#  import libraries
#  -------------------------------------------------------------------
import numpy as np
import casadi as cs
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import Polygon
from shapely import ops, affinity
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

# Get config files
#  -------------------------------------------------------------------
planning_config = sliding_pack.load_config('planning_config.yaml')
#  -------------------------------------------------------------------
def get_contact_point(beta, poseA, poseB):
    """
    Get the nearest point P between poseA and poseB.
    Compute P's coordinate's on poseA in world frame.
    """
    x, y = beta[0]/2, beta[1]/2
    base_poly = Polygon([(x, y), (-x, y), (-x, -y), (x, -y)])
    polyA = affinity.rotate(base_poly, poseA[2], origin='center', use_radians=True)
    polyB = affinity.rotate(base_poly, poseB[2], origin='center', use_radians=True)
    polyA = affinity.translate(polyA, poseA[0], poseA[1])
    polyB = affinity.translate(polyB, poseB[0], poseB[1])
    pA, _ = ops.nearest_points(polyA, polyB)

    return [pA.x, pA.y]
#  -------------------------------------------------------------------

# simulation constants
dt = 0.04

beta = [
    planning_config['dynamics']['xLenght'],
    planning_config['dynamics']['yLenght'],
    planning_config['dynamics']['pusherRadious']
]

# theta = [0.5 * np.pi, np.arctan2(0.5*beta[0], 0.5*beta[1])]
theta = [0.5 * np.pi, 0.25 * np.pi]

ctact = [[-0.5*beta[0], 0.],
         [0.5*beta[0], 0.],
         [-0.5*beta[0], -0.5*beta[1]]]

vp = [0.05, 0.01]

dyn = sliding_pack.dyn_mc.Double_Sys_sq_slider_quasi_static_ellip_lim_surf(
    planning_config['dynamics'],
    contactNum=2
)

beta, theta, ctact, vp = cs.DM(beta), cs.DM(theta), cs.DM(ctact), cs.DM(vp)

"""
# optimization variables
w = dyn.w
z = dyn.z
M = dyn.M(beta, theta, ctact)
q = dyn.q(vp)

# create quadratic programming to solve the LCP
qp = {}
qp['h'] = (2*M).sparsity()
qp['a'] = M.sparsity()
S = cs.conic('LCP2QP', 'qpoases', qp)

import pdb; pdb.set_trace()

r = S(h=2*M, g=q, a=M, lba=-q, lbx=0.)
"""

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

x_a0, x_b0 = np.array([0, 0, 0.5*np.pi]), np.array([-0.01767767, 0.41717514, 0.25*np.pi])
X_A, X_B = np.expand_dims(x_a0, axis=0), np.expand_dims(x_b0, axis=0)

while True:
    import pdb; pdb.set_trace()
    pt = get_contact_point(beta, X_A[-1, :], X_B[-1, :])

import pdb; pdb.set_trace()
