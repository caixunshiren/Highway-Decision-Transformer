import csv
import numpy as np
import ast


def parse_state(state_str):
    state = ast.literal_eval(state_str)
    state_array = np.array(state, dtype=np.float32)
    return state_array


b = np.load('dataset_5000.npy', allow_pickle=True)

for row in b:
    for state, action, reward in row:
        print(state,action,reward)

