#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fixed dependency issues

@author: nasmadasser
"""

import numpy as np


def entropy(data):   
    # count element in each category
    labels = data[:,-1]
    elements,counts = np.unique(labels,return_counts =True) 
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
            for i in range(len(elements))])
    return entropy

def information_gain(data, l_dataset, r_dataset):
    elements = data.shape[0]
    l_entropy = entropy(l_dataset)
    r_entropy = entropy(r_dataset)   
    remainder = (l_dataset.shape[0]/elements)*l_entropy + (r_dataset.shape[0] / elements) * r_entropy    
    information_gain = entropy(data)-remainder
    
    return information_gain

class DecisionTree():
    def __init__(self):
    #initialize depth
        self.depth = 0
            
    def findSplit(self, data): ## Konsti's 
        max_gain = 0
        split_attr = None
        split_val = None
        for attribute in range(data.shape[1]-1):
            #sort the dataset according to one attribute column
            data_sorted = data[np.argsort(data[:, attribute])]       
            #only compare successor values that are different to ensure that we can make the split in the tree
            for point in range(data_sorted.shape[0]-1):
                #confirm that the successor values are different to ensure that we can make the split in the tree
                if data_sorted[:point+1][-1][attribute] != data_sorted[point+1:][0][attribute]:
                    temp_gain = information_gain(data_sorted, data_sorted[:point+1], data_sorted[point+1:])
                    if temp_gain > max_gain:
                        max_gain = temp_gain
                        #value according to which the dataset is split
                        split_point = (data_sorted[:point+1][-1][attribute] + data_sorted[point+1:][0][attribute])/2
                        split_attr = attribute
                        split_val = split_point
        
        return split_attr, split_val
    
    def partition_data(self, data,split_attr,split_val):
        
        l_dataset = data[np.where(data[:,split_attr] <= split_val)]
        r_dataset = data[np.where(data[:,split_attr] > split_val)]
        
        return l_dataset,r_dataset
    
    def buildTree(self, data,depth=0):    
        one_branch ={}
        room = data[:,-1]
        
        if one_branch is None: 
            return None
        elif len(np.unique(room)) <2: # all labels same
            return None
        elif len(room) == 0:
            return None 
        
        # generate the tree
        else:
            split_attr,split_val = self.findSplit(data)
            l_dataset, r_dataset = self.partition_data(data,split_attr,split_val)
            
            l_branch = self.buildTree(l_dataset,depth+1)
            r_branch = self.buildTree(r_dataset,depth+1)
            
            one_branch = {'attribute':split_attr,'value':split_val,
            'l_branch':l_branch,'r_branch':r_branch}
            
            one_branch['depth']=depth
            
            #find the room 
                     
            #find out if this is a leaf
            if l_branch is None:
                one_branch['leaf'] = True
                one_branch['room'] = l_dataset[0][-1]
                one_branch['l_branch'] = int(l_dataset[0][-1])

                
            elif r_branch is None:
                one_branch['leaf'] = True
                one_branch['room'] = r_dataset[0][-1]
                one_branch['r_branch']= int(r_dataset[0][-1])
            else: 
                one_branch['leaf'] = False
                      
            self.depth+=1
            return one_branch

            
def predict_by_row(node, row):
    if row[node['attribute']] <= node['value']:
        if isinstance(node['l_branch'], dict):
            return predict_by_row(node['l_branch'],row)
        else: 
            return node['room']
    else:
        if isinstance(node['r_branch'],dict):
            return predict_by_row(node['r_branch'], row)
        else:
            return node['room']

def predict(X_test, trained_tree):
    prediction = []
    for row in X_test:   
        prediction.append(predict_by_row(trained_tree, row))
    return prediction
        

    


    



