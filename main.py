#!/usr/bin/env python

#Imperial College of Science and Technology London - Introduction to Machine Learning 2020 - Coursework 1


import numpy as np
from src.utils import *
from src.validation import *
from src.DecisionTree import *

__author__ = "Daphne Demekas, Nasma Dasser, Constantin Eulenstein, Kevin Landert"

if __name__ == '__main__':
    
    np.random.seed(42)
    data_clean = np.loadtxt('./data/wifi_db/clean_dataset.txt')
    data_noisy = np.loadtxt('./data/wifi_db/noisy_dataset.txt')
    print("RUNNING CLEAN DATASET")
    cross_val(data_clean,10)
    print("RUNNING NOISY DATASET")
    cross_val(data_noisy,10)