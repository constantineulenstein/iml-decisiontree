import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from IPython.display import Image
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import pydotplus
from sklearn import preprocessing
from sklearn import tree



def plot_decision_tree(clf,feature_name,target_name):
    dot_data = StringIO()  
    tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=feature_name,  
                         class_names=target_name,  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())
    return graph
    


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
        return dataset,train,test

dataset,train,test=load().load_data('clean')

def plot_benchmark():
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    train = pd.DataFrame(train)
    X_train,y_train = train.iloc[:,:-1], train.iloc[:,-1]
    clf = clf.fit(X_train,y_train)
    graph=plot_decision_tree(clf, X_train.columns, ['room1','room2','room3','room4'])
    graph.write_png("decision_tree.png")