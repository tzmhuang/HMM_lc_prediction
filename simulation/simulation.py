import os
import sys
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import sympy


class Polynomial(object):
    def __init__(self, c0=0, c1=0, c2=0, c3=0, c4=0, c5=0):
        _x = sympy.symbols('x')
        self.expr = c0 + c1*_x + c2*_x**2 + c3*_x**3 + c4*_x**4 + c5*_x**5
        return

    def eval(self, x, order=0):
        expr = self.expr
        for i in range(order):
            expr = sympy.diff(expr)
        return expr.subs(_x, x)


class Bezier(object):
    def __init__(self, pt1, pt2, pt3, pt4):
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3
        self.pt4 = pt4
        return

    def eval(self, t, order):
        if order == 0:
            x = (1-t)**3*self.pt1[0] + 3*(1-t)**2*t*self.pt2[0] + \
                3*(1-t)*t**2*self.pt3[0] + t**3*self.pt4[0]
            y = (1-t)**3*self.pt1[1] + 3*(1-t)**2*t*self.pt2[1] + \
                3*(1-t)*t**2*self.pt3[1] + t**3*self.pt4[1]
        elif order == 1:
            x = -3*(1-t)**2*self.pt1[0] + 3*(t-1)*(3*t-1)*self.pt2[0] + \
                3*(2*t-3*t**2)*self.pt3[0] + 3*t**2*self.pt4[0]
            y = -3*(1-t)**2*self.pt1[1] + 3*(t-1)*(3*t-1)*self.pt2[1] + \
                3*(2*t-3*t**2)*self.pt3[1] + 3*t**2*self.pt4[1]
            # print(self.pt1[0], self.pt2[0], x, y)
        else:
            raise("Invalid order")
        return x, y

    def get_coeff(self, order):
        x1 = np.array(self.pt1)
        x2 = np.array(self.pt2)
        x3 = np.array(self.pt3)
        x4 = np.array(self.pt4)
        c1 = -x1 + 3*x2 - 3*x3 + x4
        c2 = 3*x1 - 6*x2 + 3*x3
        c3 = -3*x1 + 3*x2
        c4 = x1
        if order == 0:
            return c1, c2, c3, c4
        elif order == 1:
            c1 = 3*c1
            c2 = 2*c2
            return c1, c2, c3
        else:
            raise("Invalid order")


class BezierTrajectory(object):
    def __init__(self, bezier, speed):
        self.bezier = bezier
        self.speed = speed

    def set_ts(self, dt):
        ts = [0]
        while ts[-1] <= 1:
            dx, dy = self.bezier.eval(ts[-1], 1)
            vl = np.sqrt(dx**2+dy**2)
            l = dt*self.speed
            ts += [ts[-1]+l/vl]
        self.ts = np.array(ts)

    def get_heading(self):
        dx, dy = self.bezier.eval(self.ts, 1)
        vx = dx/np.sqrt(dx**2 + dy**2) * self.speed
        vy = dy/np.sqrt(dx**2 + dy**2) * self.speed
        return np.arctan2(dy, dx)

    def get_coord(self):
        return self.bezier.eval(self.ts, 0)

    def get_speed(self):
        h = self.get_heading()
        vx = np.cos(h) * self.speed
        vy = np.sin(h) * self.speed
        return self.speed, vx, vy


class Road(object):
    def __init__(self, centre_function, length=100, half_lane_width=2, lane_number=3):
        self.center_function = centre_function  # Center Function
        self.l = length
        self.hlw = half_lane_width
        self.lane_number = lane_number
        return

    def get_dist_to_lane_center(self, pt):
        x = pt[0]
        y = pt[1]
        dist_to_centre = y - self.center_function(x)
        sign = 2*(dist_to_centre >= 0) - 1
        if int(abs(dist_to_centre)/self.hlw % 2) > 0:
            d = sign*(abs(dist_to_centre) % self.hlw - self.hlw)
        else:
            d = sign*(abs(dist_to_centre) % self.hlw)
        return d


class Vehicle(object):
    def __init__(self, dt, trajectory, start=0, speed=10, width=2, length=3, x=0, y=0, dh=0):
        self.s = start
        self.v = speed
        self.w = width
        self.l = length
        self.x = x
        self.y = y
        self.h = math.atan2(trajectory.eval(s, 1), 1)
        self.dt = dt
        self.dh = dh
        self.traj = trajectory
        return

    def update():
        self.s += self.v*self.dt
        self.x, self.y = self.traj.eval(self.s, 0)
        dtraj = self.traj.eval(self.s, 1)
        self.h = math.atan2(dtraj[0], dtraj[1])


class Ego(Vehicle):
    def get_obstacle(self, vehicle):
        return

    pass


class Simulation(object):
    def __init__(self, SimTime=100, **kwargs):
        self.road = kwargs['road']
        self.ego = kwargs['ego']
        self.vehicles = kwargs['road_vehicles']
        self.simT = SimTime
        return
