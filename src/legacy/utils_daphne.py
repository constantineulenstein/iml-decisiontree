from load_data import *
import os
import numpy as np 
from utils_kevin import *



def find_split(train_data):
    x_data = np.delete(train_data, -1, axis=1)
    minimum = int(np.min(x_data))
    maximum = int(np.max(x_data))
    
    biggest_gain = 0
    best_split = []
    best_pair = []

    for attr in range(x_data.shape[1]):
        for value in range(minimum,maximum):
            split = partition_data(train, [attr,value])
            info_gain = information_gain(train_data,split[0],split[1])
            if info_gain > biggest_gain:
                biggest_gain = info_gain
                best_split = split
                best_pair = [attr,value]
    
    return best_split[0], best_split[1], best_pair, biggest_gain
