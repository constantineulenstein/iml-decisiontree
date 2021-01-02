#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:44:24 2020

@author: nasmadasser
"""

import numpy as np 
import matplotlib.pyplot as plt

def confusion_matrix(true, predicted):
    # set the length of the matrix
    K = len(np.unique(true))+1
    # create empty matrix to fill
    matrix = np.zeros((K,K))
    
    # count labels 
    for a, p in zip(true, predicted):
        matrix[a][p] +=1
    
    return matrix[1:K,1:K]


def plot_confusion_matrix(cm, title= 'Confusion matrix', cmap=plt.cm.winter_r):
    plt.figure(figsize=(15,15))    
    plt.matshow(cm, cmap=cmap) # imshow
    plt.colorbar()
    plt.xlabel('predicted room')
    plt.ylabel('actual room')
    plt.tight_layout()
    
    height, width = cm.shape
    for x in range (width):
        for y in range (height):
            plt.annotate(str(cm[x][y]), xy=(y,x), ha='center', va='center')
    plt.show()

def accuracy(true, predicted):
    return (true == predicted).sum()/float(len(true))

def precision(room, cm):
    # precision = TP / (TP+FP)
    column = cm[:,room]
    return cm[room,room]/column.sum()

def recall(room, cm):
    # recall = TP/ (FN+TP)
    row= cm[room,:]
    return cm[room,room]/row.sum()

def precision_total(cm):
    rows, columns = cm.shape
    precisions =0
    for room in range(rows):
        precisions +=precision(room, cm)
    return precisions/rows

def recall_total(cm):
    rows, columns = cm.shape
    recalls=0
    for room in range(columns):
        recalls  += recall(room, cm)
    return recalls/columns

def cm_precision_recall_accuracy_F1(y_test, predicted, cm, by_room):
    precision =precision_total(cm)
    recall = recall_total(cm)
    accura = accuracy(y_test, predicted)
    F1 = 2*(precision*recall)/(precision+recall)    
    if by_room is True:
        print('\nPrediction by room \n', " Room  Precision   Recall")
        for room in range(4):
            print(f"{room:5d} {precision:9.3f} {recall:6.3f}")
    
    print ('\nTotal precision', "{:.3%}".format(precision))
    print ('Total recall', "{:.3%}".format(recall))
    print('Total accuracy',"{:.3%}".format(accura))
    print('F1 score',"{:.3%}".format(F1))
    
    return precision, recall, accura, F1
    