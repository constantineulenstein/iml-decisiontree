#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:37:52 2020

@author: nasmadasser
"""
from decision_tree_nasma import *
from validation_nasma import *

## few attempts to visualize
def pretty_print_tree(node, depth =0):
	if isinstance(node, dict):
		print('%s signal%s : %.50s' % ((depth*'\t', (node['attribute']+1), node['value'])))
		pretty_print_tree(node['l_branch'], depth+1)
		pretty_print_tree(node['r_branch'], depth+1)
	else:
		print('%s%s' % ((depth*'-', node)))

def visualize_tree(node, file=None, _prefix="", _last=True):
    if isinstance(node,dict):
        print(_prefix, "`-- " if _last else "|-- ", node['value'], sep="", file=file)
        _prefix += "   " if _last else "|  "
        child_count = len(node['l_branch'])
        for i, child in enumerate(node['l_branch']):
            _last = i == (child_count - 1)
        visualize_tree(child, file, _prefix, _last)

def visualize(node,prefix=""):
    if isinstance(node['l_branch'],dict):
        print(prefix, "|--" , node['value'], 'depth',node['depth'])

        prefix+= " "
        visualize(node['l_branch'],prefix)

    else:
        print(prefix*node['depth'],"|--",node['value'],"\n" ,
              prefix*node['depth'],'`--','Room',node['l_branch'],'depth',node['depth'])
        
# split and get the metrics     
dataset=load_data('clean')
train, test = train_test_split(data=dataset, ratio =0.8)
trained_tree =DecisionTree().buildTree(train) 

visualize(trained_tree)

