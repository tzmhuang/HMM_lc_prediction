import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import simulation
import random
import json
import datetime


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "data/")
DATA_FILE = os.path.join(SAVE_DIR, "20200304_192852.json")

with open(DATA_FILE, 'r') as f:
    string = f.read()
    data = json.loads(string)

# Finding Switching point


def label(fit_n, write):
    for k in data.keys():
        dtc = np.array(data[k]['dist_to_center'])
        sidx = abs(np.diff(dtc)).argmax()+1
        coef_before = np.polyfit(
            data[k]['time'][sidx-fit_n:sidx], dtc[sidx-fit_n:sidx], 1)
        coef_after = np.polyfit(
            data[k]['time'][sidx:sidx+fit_n], dtc[sidx:sidx+fit_n], 1)
        before_t = -1*coef_before[1]/coef_before[0]
        after_t = -1*coef_after[1]/coef_after[0]
        print(coef_before, coef_after)
        print(before_t, after_t)
        data[k]['label'] = np.zeros(dtc.shape)
        data[k]['label'][(np.array(data[k]['time']) > before_t)
                         * (np.array(data[k]['time']) < after_t)] = 1
        data[k]['label'] = data[k]['label'].tolist()
    if write:
        with open(DATA_FILE+'.labeled', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    label(10, True)

    fig1, ax1 = plt.subplots()
    for k in data.keys():
        dist = data[k]['dist_to_center']
        x = data[k]['x']
        lb = data[k]['label']
        ax1.plot(data[k]['time'], dist, 'x')
        ax1.plot(data[k]['time'], lb, 'r')

        dtc = np.array(data[k]['dist_to_center'])
        sidx = abs(np.diff(dtc)).argmax()+1
        ax1.plot(data[k]['time'][sidx-10:sidx],
                 dtc[sidx-10:sidx], 'o')
        ax1.plot(data[k]['time'][sidx:sidx+10],
                 dtc[sidx:sidx+10], '+')

    plt.show()
