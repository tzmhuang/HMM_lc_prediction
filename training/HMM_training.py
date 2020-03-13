import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from HiddenMarkovModel import HiddenMarkovModel

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_FILE = os.path.join(
    PROJECT_ROOT, "../simulation/data/20200304_192852.json.labeled")

with open(DATA_FILE, 'r') as f:
    string = f.read()
    data = json.loads(string)


# Initialize A and B Matrix
states = {'left': 0, 'none': 1, 'right': 2}

model = HiddenMarkovModel(states)
model.initialize()


# fig, axs = plt.subplots(7, 7, sharex='col', sharey='row')
# for k in data.keys():
#     key_list = [i for i in list(data[k].keys())
#                 if i != 'speed' and i != 'label']
#     for idx, e in enumerate(key_list):
#         for idxx, ee in enumerate(key_list):
#             axs[idx][idxx].plot(data[k][ee], data[k][e], '.')
#             axs[idx][idxx].set_xlabel(ee)
#             axs[idx][idxx].set_ylabel(e)
#     fig.suptitle(k)
#
# plt.show()
