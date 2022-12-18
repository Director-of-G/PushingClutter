#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version Copyright (c) 2010 kne / sirkne at gmail dot com
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

#  import libraries
#  -------------------------------------------------------------------
from Box2D.examples.framework import (Framework, Keys, main)
from Box2D import (b2FixtureDef, b2PolygonShape,
                   b2Transform, b2Mul, b2Vec2,
                   b2_pi)
#  -------------------------------------------------------------------
import casadi as cs
from copy import deepcopy
from math import sqrt
import numpy as np
from scipy.spatial import KDTree
import time
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------


def convert_vec2array(vec):
    return np.array([vec[0], vec[1]])


class SliderStatus(object):
    def __init__(self, handle) -> None:
        self.handle = handle

        self.x = 0.
        self.y = 0.
        self.theta = 0.
        self.psi = 0.
        self.update()

    def state(self):
        return [self.x, self.y, self.theta, self.psi]

    def update(self, psi=None):
        pos = convert_vec2array(self.handle.position)
        self.x = pos[0]
        self.y = pos[1]
        self.theta = self.handle.angle

        if psi is not None:
            self.psi = psi

    def update_psi(self, psi):
        self.psi = psi


class Controller(object):
    """
    Model predictive controller for planar pushing.
    """
    def __init__(self, scale=10.0) -> None:
        # Get config files
        self.tracking_config = sliding_pack.load_config('tracking_config.yaml')
        self.planning_config = sliding_pack.load_config('nom_config.yaml')
        
        # Set problem constants
        self.scale = scale
        self.T = 10
        self.freq = 25
        self.N_MPC = 25
        self.dt = 1.0/self.freq
        self.N = int(self.T*self.freq)
        self.Nidx = int(self.N)
        self.beta_ = [
            self.planning_config['dynamics']['xLenght'],
            self.planning_config['dynamics']['yLenght'],
            self.planning_config['dynamics']['pusherRadious']
        ]
        
        # Define system dynamics
        self.dynTrack = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
            self.tracking_config['dynamics'],
            self.tracking_config['TO']['contactMode']
        )
        self.dynPlan = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
            self.planning_config['dynamics'],
            self.planning_config['TO']['contactMode']
        )
        self.optObj = None

        self.X_nom_val = None
        self.X_nom_tree = None  # KDTree for nearest point query
        self.U_nom_val_opt = None
        self.u = [0., 0., 0.]  # default input

        # plan nominal trajectory     
        self.generate_nominal_trajectory()

        self.t0 = time.clock()

    def beta(self):
        return self.beta_

    def generate_nominal_trajectory(self):
        x0_nom, x1_nom = sliding_pack.traj.generate_traj_sine(0.0, 1.0, 0.5, 0.05, self.N, self.N_MPC)
        self.X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, self.dt)
        self.X_nom_tree = KDTree(data=self.X_nom_val[:2, :].T)

        # Solve the planning problem
        optObjPlan = sliding_pack.to.buildOptObj(
            self.dynPlan, self.N+self.N_MPC,
            self.planning_config['TO'], dt=self.dt)
        resultFlag, X_nom_val_opt, U_nom_val_opt, _, _, _ = optObjPlan.solveProblem(
            0, [0., 0., 0.*(np.pi/180.), 0.], self.beta_,
            X_warmStart=self.X_nom_val)
        if self.dynTrack.Nu > self.dynPlan.Nu:
            U_nom_val_opt = cs.vertcat(
                    U_nom_val_opt,
                    cs.DM.zeros(np.abs(self.dynTrack.Nu - self.dynPlan.Nu), self.N+self.N_MPC-1))
        elif self.dynPlan.Nu > self.dynTrack.Nu:
            U_nom_val_opt = U_nom_val_opt[:self.dynTrack.Nu, :]

        self.U_nom_val_opt = U_nom_val_opt

        # Define the optimization problem for tracking
        self.optObj = sliding_pack.to.buildOptObj(
            self.dynTrack, self.N_MPC,
            self.tracking_config['TO'],
            self.X_nom_val, self.U_nom_val_opt, dt=self.dt,
        )

    def nearest_point_index(self, query):
        """
        Return the index of the nominal point which is the nearest to query.
        """
        _, index = self.X_nom_tree.query(query)

        return index

    def solve_mpc(self, x_init):
        """
        Solve one step model predictive control
        """
        if time.clock() - self.t0 >= 20*self.dt:
            index = self.nearest_point_index(x_init[:2])
            resultFlag, x_opt, u_opt, del_opt, f_opt, t_opt = self.optObj.solveProblem(
                index, x_init, self.beta_,
                S_goal_val=None,
                obsCentre=None, obsRadius=None)
            u0 = u_opt[:, 0].elements()
            x0 = (x_init + self.dynTrack.f(x_init, u0, self.beta_)*self.dt).elements()
            u = [u0[0], u0[1], x0[3]]
            self.u = deepcopy(u)
            self.t0 = time.clock()
        else:
            u = deepcopy(self.u)

        return u  # (fn, ft, psi)


class PusherSlider(Framework):
    name = "PusherSlider"
    description = "Planar pushing simulation for single slider."

    def __init__(self, numObj=1, scale=10.0):
        """
        :param numObj: the number of sliders.
        :param scale: dimension scale in Box2d simulation.
        """
        super(PusherSlider, self).__init__()
        self.body_dict = {}
        self.slider_status = {}
        self.numObj = numObj

        self.world.gravity = (0.0, 0.0)

        self.controller = Controller()
        self.geom_beta = self.controller.beta()
        self.scale = scale

        # The boundaries
        halflen = 2.0 * self.scale
        ground = self.world.CreateBody(position=(0, 0))
        ground.CreateEdgeChain(
            [(-halflen, -halflen),
             (-halflen, halflen),
             (halflen, halflen),
             (halflen, -halflen),
             (-halflen, -halflen)]
        )

        self.body_dict['ground'] = ground

        gravity = 10.0
        self.box_size = [self.geom_beta[0]*self.scale, self.geom_beta[1]*self.scale]
        fixtures = b2FixtureDef(shape=b2PolygonShape(box=(self.box_size[0], self.box_size[1])),
                                density=1, friction=0.2)

        for i in range(self.numObj):
            body = self.world.CreateDynamicBody(
                position=(0, 0 + 1.0 * i), fixtures=fixtures)

            self.body_dict['slider{0}'.format(i+1)] = body
            self.slider_status['slider{0}'.format(i+1)] = SliderStatus(body)

            # For a circle: I = 0.5 * m * r * r ==> r = sqrt(2 * I / m)
            r = sqrt(2.0 * body.inertia / body.mass)

            self.world.CreateFrictionJoint(
                bodyA=ground,
                bodyB=body,
                localAnchorA=(0, 0),
                localAnchorB=(0, 0),
                collideConnected=True,
                maxForce=0.1*body.mass * gravity,
                maxTorque=0.1*body.mass * r * gravity
            )

    def Step(self, settings):
        # update 
        for key in self.slider_status:
            self.slider_status[key].update()
        x_init = self.slider_status['slider1'].state()
        x_init_scaled = [x_init[0]/self.scale, x_init[1]/self.scale] + x_init[2:]
        # solve mpc
        try:
            # u = self.controller.solve_mpc(x_init_scaled)
            u = [0., 0., 0.]
        except:
            import pdb; pdb.set_trace()
        xc = -self.box_size[0]/2+0.01; yc = -(self.box_size[0]/2)*np.tan(u[2])
        f = self.body_dict['slider1'].GetWorldVector(localVector=(4.5, -0.9))
        yc = -0.5
        p = self.body_dict['slider1'].GetWorldPoint(localPoint=(xc, yc))
        self.body_dict['slider1'].ApplyForce(f, p, True)
        self.slider_status['slider1'].update_psi(u[2])
        print('Applied force ({0}) at point ({1})'.format(f, p))

        super(PusherSlider, self).Step(settings)

if __name__ == "__main__":
    main(PusherSlider)
