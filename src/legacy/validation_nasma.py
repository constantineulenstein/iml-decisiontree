#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:37:03 2020

@author: nasmadasser
"""

import random 
import numpy as np
import matplotlib.pyplot as plt
from decision_tree_nasma import *
from metrics_nasma import *

def load_data(clean_or_noisy:str): 
    # get the path to the chosen
    data_path = '../data/wifi_db/'
    dataset = np.loadtxt(data_path + clean_or_noisy+'_dataset.txt')
    return dataset

def train_test_split(data, ratio):    
    ratio = int(data.shape[0]*ratio)
    # randomly shuffle 
    np.random.seed(23)
    np.random.shuffle(data)  
    # we apply 80/20
    train,test   = data[:ratio,:], data[ratio:,:]
    return train, test

# k- fold crossvalidations 
def cross_validation_split(data, k_folds):
    data_split =[]
    data_copy = list(data)
    fold_size = int(len(data)/k_folds)
    for k in range (k_folds):
        fold =[]
        while len(fold) < fold_size:
            index = random.randrange(len(data_copy))
            fold.append(data_copy.pop(index))
        data_split.append(fold)
    return data_split    

def evaluate_train_test_split(test, trained_tree): 
    X_test = test[:,:-1]
    y_test = test[:,-1].astype(int)
    predicted = predict(X_test,trained_tree)
    predicted = np.asarray(predicted).astype(int)
    cm =confusion_matrix(y_test, predicted)
    cm_precision_recall_accuracy_F1(y_test, predicted, cm)
    plot_confusion_matrix(cm)   

    
def k_cross_validation(dataset, k_folds=10):
    #def evaluate_k_fold(data, k_folds, *args):
    np.random.seed(23)
    np.random.shuffle(dataset)  
    folds = np.array_split(dataset,k_folds)
    
    matrix = []
    scores =[]
    for i in range(k_folds):
        train_k = folds.copy()
        test_k = folds[i]
        del train_k[i]
        print('\n###### FOLD ',i, '##############')
        train_k = np.vstack(train_k)
        trained_tree = DecisionTree().buildTree(train_k.copy()) 
        X_test = test_k[:,:-1]
        y_test = test_k[:,-1]
        predicted = predict(X_test.copy(),trained_tree)
        predicted = np.asarray(predicted).astype(int)
        y_test = y_test.astype(int)
        cm = confusion_matrix(y_test.copy(),predicted)
        precis, recal, accura, F1=cm_precision_recall_accuracy_F1(y_test, predicted, cm, by_room=False)
        matrix.append(cm)
        scores.append((precis, recal, accura, F1))
    
    scores= np.vstack(scores)
    avg_scores = np.mean(scores, axis=0)
    matrix= np.mean(matrix, axis=0)
    
    print ('\n## RESULTS OF K-FOLD ##\n',
           'avg precision: {:.3%}\n'.format(avg_scores[0]),
           'avg recall: {:.3%}\n'.format(avg_scores[1]),
           'avg accuracy: {:.3%}\n'.format(avg_scores[2]),
           'avg F1 Score: {:.3%}\n'.format(avg_scores[3]),
           'CONFUSION MATRIX {}\n' .format(np.matrix(matrix)))
    
    recall_room=[]
    precision_room = []
    # by room
    for room in range(4):
        recall_room.append(recall(room,cm))
        precision_room.append(precision(room,cm))
        print('room  recall precision')
        print(room, "{:.2%}".format(recall(room,cm)), "{:.2%}".format(precision(room,cm)))
        
        
  ##########  for the main one    

# split and get the metrics     
dataset=load_data('clean')
train, test = train_test_split(data=dataset, ratio =0.8)

### to test and ge tthe confusion matrix 
trained_tree =DecisionTree().buildTree(train) 


## k-fold cross validation
k_cross_validation(dataset, k_folds=10)     

## normal validation
#evaluate(test, trained_tree)