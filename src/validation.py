import numpy as np
from src.DecisionTree import DecisionTree
from src.utils import *
import datetime



##############################################################################
#Does crossvalidation
#Inputs:
#data: dataset to do crossvalidation on
#folds: number of folds
    
#Returns: 
#Tuples: Statistics for unpruned and pruned version

##############################################################################
def cross_val(data,folds):
    
    
    start = datetime.datetime.now()
    
    
    splits = crossValidation_split(data,folds)
    stats = []
    stats_pruned = []
    
    #Creats a tree and evaluates for every fold. Results collected in list "stats"
    for i in range(folds):
        
        #Take one testset
        test = data[splits[i],:]
        results_lowlevel_unpruned = []
        results_lowlevel_pruned = []
        
        #Take only non-test folds for train and val
        remaining_folds = np.delete(np.arange(folds),i)
        for j in remaining_folds:
            
            
            mask_train = np.ones(data.shape[0],dtype = bool)
            mask_train[splits[i]] = False
            mask_train[splits[j]] = False
            
            train = data[mask_train, :]
            val = data[splits[j]]
            
            decisiontree = DecisionTree()
            decisiontree.buildTree(train)
            
            results_lowlevel_unpruned.append(decisiontree.evaluate(test))
            
            decisiontree.prune(train,val)
            results_lowlevel_pruned.append(decisiontree.evaluate(test))
            print("Tree Complete! Test Set: {} Validation Set: {}".format(i,j))
            
        stats.append(average_statistics(results_lowlevel_unpruned))
        stats_pruned.append(average_statistics(results_lowlevel_pruned))
    
    result = average_statistics(stats)
    result_pruned = average_statistics(stats_pruned)
    
    end = datetime.datetime.today()
    
    
    
    print("")
    print("#############################")
    print("{}-fold Crossvalidation complete. Average scores unpruned tree over all folds:".format(folds))
    print("Average Confusion Matrix:")
    print(result['confusionmatrix'])
    print("Average Precision: {:.2%}".format(np.mean(result['precision'])))
    print("Average Precision per Class:")
    print(result['precision'])
    print("Average Recall: {:.2%}".format(np.mean(result['recall'])))
    print("Average Recall per Class:")
    print(result['recall'])
    print("Average F1: {:.2%}".format(np.mean(result['F1score'])))
    print("Average F1 score per Class:")
    print(result['F1score'])
    print("Average Classification Rate: {:.2%}".format(result['posClassRate']))
    print("#############################")
    print("Average scores pruned tree over all folds:")
    print("Average Confusion Matrix:")
    print(result_pruned['confusionmatrix'])
    print("Average Precision: {:.2%}".format(np.mean(result_pruned['precision'])))
    print("Average Precision per Class:")
    print(result_pruned['precision'])
    print("Average Recall: {:.2%}".format(np.mean(result_pruned['recall'])))
    print("Average Recall per Class:")
    print(result_pruned['recall'])
    print("Average F1 score: {:.2%}".format(np.mean(result_pruned['F1score'])))
    print("Average F1 score per Class:")
    print(result_pruned['F1score'])
    print("Average Classification Rate: {:.2%}".format(result_pruned['posClassRate']))
    print("Runtime: {}".format(end - start))
    print("#############################")
    print("")
    
    return result , result_pruned


#Get average for every measure across folds
def average_statistics(stats):
    
    result = {'confusionmatrix' : np.mean(getmeasure(stats,'confusionmatrix'), axis = 0),
              'precision' : np.mean(getmeasure(stats,'precision'), axis = 0),
              'recall' : np.mean(getmeasure(stats,'recall'), axis = 0),
              'F1score': np.mean(getmeasure(stats,'F1score'), axis = 0),
              'posClassRate' : np.mean(getmeasure(stats,'posClassRate'), axis = 0) }
    return result
    