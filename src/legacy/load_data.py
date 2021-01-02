#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:21:28 2020

@author: nasmadasser
"""

import numpy as np

class load:
    def load_data(self,clean_or_noisy:str): 
        # get the path to the chosen
        data_path = '../data/wifi_db/'
        dataset = np.loadtxt(data_path + clean_or_noisy+'_dataset.txt')
        
        # randomly shuffle 
        np.random.seed(23)
        np.random.shuffle(dataset)
        
        # we apply 80/20
        train,test   = dataset[:1600,:], dataset[1600:,:]
        #X_train,y_train = train[:,:-1], train[:,-1]
        #X_test,y_test   = test[:,:-1], test[:,-1]
        return dataset,train,test