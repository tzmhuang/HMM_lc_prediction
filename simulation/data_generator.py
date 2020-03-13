import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import simulation
import random
import scipy.stats as stats
import codecs
import json
import datetime

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "data/")

DATA_NUM = 2
RLC_NUM = int(DATA_NUM/2)
LLC_NUM = DATA_NUM - RLC_NUM
SEED = 2020
SD_RANGE = 2
LANE_WIDTH = 4
AVG_LANECHANGE_LENGTH = 100
AVG_SPEED = 20

FLAG_DUMP_JSON = True
FLAG_VISUALIZE = True

# Left
pt1l_y_dist = stats.truncnorm(-SD_RANGE, SD_RANGE, 0, 0.25)
pt1l_x = np.zeros((LLC_NUM, 1))
pt1l_y = pt1l_y_dist.rvs(LLC_NUM).reshape((-1, 1))
pt1l = np.concatenate((pt1l_x, pt1l_y), 1)

pt2l_x_dist = stats.truncnorm(-SD_RANGE, 0, AVG_LANECHANGE_LENGTH/2, 10)
pt2l_y_dist = stats.truncnorm(-SD_RANGE, SD_RANGE, 0, 0.25)
pt2l_x = pt2l_x_dist.rvs(LLC_NUM).reshape((-1, 1))
pt2l_y = pt2l_y_dist.rvs(LLC_NUM).reshape((-1, 1))
pt2l = np.concatenate((pt2l_x, pt2l_y), 1)

pt3l_x_dist = stats.truncnorm(0, SD_RANGE, AVG_LANECHANGE_LENGTH/2, 10)
pt3l_y_dist = stats.truncnorm(-SD_RANGE, SD_RANGE, LANE_WIDTH, 0.25)
pt3l_x = pt3l_x_dist.rvs(LLC_NUM).reshape((-1, 1))
pt3l_y = pt3l_y_dist.rvs(LLC_NUM).reshape((-1, 1))
pt3l = np.concatenate((pt3l_x, pt3l_y), 1)

pt4l_x_dist = stats.truncnorm(
    (max(pt3l_x)-AVG_LANECHANGE_LENGTH)/10, SD_RANGE, AVG_LANECHANGE_LENGTH, 10)
pt4l_y_dist = stats.truncnorm(-SD_RANGE, SD_RANGE, LANE_WIDTH, 0.25)
pt4l_x = pt4l_x_dist.rvs(LLC_NUM).reshape((-1, 1))
pt4l_y = pt4l_y_dist.rvs(LLC_NUM).reshape((-1, 1))
pt4l = np.concatenate((pt4l_x, pt4l_y), 1)

# Right
pt1r_y_dist = stats.truncnorm(-SD_RANGE, SD_RANGE, 0, 0.25)
pt1r_x = np.zeros((RLC_NUM, 1))
pt1r_y = pt1r_y_dist.rvs(RLC_NUM).reshape((-1, 1))
pt1r = np.concatenate((pt1r_x, pt1r_y), 1)

pt2r_x_dist = stats.truncnorm(-SD_RANGE, 0, AVG_LANECHANGE_LENGTH/2, 10)
pt2r_y_dist = stats.truncnorm(-SD_RANGE, SD_RANGE, 0, 0.25)
pt2r_x = pt2r_x_dist.rvs(RLC_NUM).reshape((-1, 1))
pt2r_y = pt2r_y_dist.rvs(RLC_NUM).reshape((-1, 1))
pt2r = np.concatenate((pt2r_x, pt2r_y), 1)

pt3r_x_dist = stats.truncnorm(0, SD_RANGE, AVG_LANECHANGE_LENGTH/2, 10)
pt3r_y_dist = stats.truncnorm(-SD_RANGE, SD_RANGE, -LANE_WIDTH, 0.25)
pt3r_x = pt3r_x_dist.rvs(RLC_NUM).reshape((-1, 1))
pt3r_y = pt3r_y_dist.rvs(RLC_NUM).reshape((-1, 1))
pt3r = np.concatenate((pt3r_x, pt3r_y), 1)

pt4r_x_dist = stats.truncnorm(
    (max(pt3r_x)-AVG_LANECHANGE_LENGTH)/10, SD_RANGE, AVG_LANECHANGE_LENGTH, 10)
pt4r_y_dist = stats.truncnorm(-SD_RANGE, SD_RANGE, -LANE_WIDTH, 0.25)
pt4r_x = pt4r_x_dist.rvs(RLC_NUM).reshape((-1, 1))
pt4r_y = pt4r_y_dist.rvs(RLC_NUM).reshape((-1, 1))
pt4r = np.concatenate((pt4r_x, pt4r_y), 1)


# Speed
speed_dist = stats.truncnorm(-SD_RANGE, SD_RANGE, AVG_SPEED, 5)
speed = speed_dist.rvs(DATA_NUM)

# Generate Bezier Trajectory
lctraj_left = []
for i in range(LLC_NUM):
    bc = simulation.Bezier(pt1l[i], pt2l[i], pt3l[i], pt4l[i])
    traj = simulation.BezierTrajectory(bc, speed[i])
    lctraj_left += [traj]

lctraj_right = []
for i in range(RLC_NUM):
    bc = simulation.Bezier(pt1r[i], pt2r[i], pt3r[i], pt4r[i])
    traj = simulation.BezierTrajectory(bc, speed[LLC_NUM+i])
    lctraj_right += [traj]

lctraj = lctraj_left + lctraj_right

# Define road


def center_func(x): return 0


road = simulation.Road(
    center_func, half_lane_width=LANE_WIDTH/2, length=150)


# Calculate Data
traj_dict = {}
delta_t = 0.05
for idx, traj in enumerate(lctraj):
    traj.set_ts(delta_t)
    print("TS", traj.ts)
    print("Speed", traj.speed)
    x, y = traj.get_coord()
    heading = traj.get_heading()
    speed, vx, vy = traj.get_speed()
    # print("D", np.sqrt(np.diff(x)**2 + np.diff(y)**2)/speed)
    # print("Dx", np.diff(x)/vx[0:-1])
    # print("Dy", np.diff(y)/vy[0:-1])
    dist_to_center = []
    t_list = []
    t = 0
    for i in range(x.size):
        pt = [x[i], y[i]]
        dist_to_center += [road.get_dist_to_lane_center(pt)]
        t_list += [t]
        t += delta_t

    data = {'time': t_list, 'x': x.tolist(), 'y': y.tolist(),
            'speed': speed, 'vx': vx.tolist(), 'vy': vy.tolist(),
            'heading': heading.tolist(), 'dist_to_center': dist_to_center}
    traj_dict[str(idx)] = data


# Dump json
if FLAG_DUMP_JSON:
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    file_name = os.path.join(
        SAVE_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.json')
    if os.path.exists(file_name):
        raise ("File already exists")
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(traj_dict, f, ensure_ascii=False, indent=2)

# Visualize
if FLAG_VISUALIZE:
    fig2, ax2 = plt.subplots()
    for k in traj_dict.keys():
        dist = traj_dict[k]['dist_to_center']
        x = traj_dict[k]['x']
        ax2.plot(x, dist, 'x')
        # print("Dist", dist)

    fig1, ax1 = plt.subplots()
    for traj in lctraj:
        x, y = traj.get_coord()
        dx, dy = traj.bezier.eval(traj.ts, 1)
        ax1.quiver(x, y, dx, dy)
        ax1.axis('equal')

    plt.plot(np.linspace(0, road.l, 1000), np.ones(1000)*road.hlw, 'k')
    plt.plot(np.linspace(0, road.l, 1000), np.ones(1000)*-road.hlw, 'k')
    plt.plot(np.linspace(0, road.l, 1000), np.ones(1000)*road.hlw*3, 'k')
    plt.plot(np.linspace(0, road.l, 1000), np.ones(1000)*-road.hlw*3, 'k')

    plt.plot(np.linspace(0, road.l, 1000), np.zeros(1000), 'k--')
    plt.plot(np.linspace(0, road.l, 1000), np.ones(1000)*road.hlw*2, 'k--')
    plt.plot(np.linspace(0, road.l, 1000),
             np.ones(1000)*-road.hlw*2, 'k--')
    plt.show()
