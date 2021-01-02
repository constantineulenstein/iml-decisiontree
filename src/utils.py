#Contains various helper functions used by DecisionTree and validation

import numpy as np
from functools import reduce
import matplotlib.pyplot as plt


def get_xydata(data):
    return data[:,:-1] , data[:,-1]

#Used for testing
def load_testdata(name : str ):
    
    basepath = '../data/testing_data/'
    path = basepath + name
    data = np.loadtxt(path)

    return data

#Used for averaging statistics measures
def getmeasure(stats,measure):
    return [dic[measure] for dic in stats]


#Computest the information gain
def information_gain(data_all,data_left,data_right):
    
    cnt_all = data_all.shape[0]
    cnt_left = data_left.shape[0]
    cnt_right = data_right.shape[0]
    
    remainder = cnt_left/cnt_all * entropy(data_left) + cnt_right / cnt_all * entropy(data_right)
    gain = entropy(data_all) - remainder
    
    return gain

#Computes the entropy
def entropy(data):
    
    cnt = data.shape[0]
    _,counts = np.unique(data[:,-1],return_counts = True)
    return - sum(counts/cnt * np.log2(counts/cnt))


#Split data for crossvalidation
def crossValidation_split(data,folds):
    
    np.random.seed(42)
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    splits = np.array_split(index,folds)
    return splits
  
#Load data in Jupyternotebook    
def get_data(name: str):
    
    np.random.seed(10)
    basepath = '../data/wifi_db/'
    path = basepath + name
    data = np.loadtxt(path)
    np.random.shuffle(data)
    return data

#Plot the averaged final confusion matrix 
def plot_confusion_matrix(cm, title, cmap=plt.cm.gray_r):
    
    fig=plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    # imshow
    cax=ax.matshow(cm, cmap=cmap) 

    # set the correct labels
    alpha = ['1','2','3','4']
    ax.set_xticklabels(['']+alpha,fontsize=12)
    ax.set_yticklabels(['']+alpha,fontsize=12)
    height, width = cm.shape
    for x in range (width):
        for y in range (height):
            # color of the diagonal different than the others
            if x == y:           
                plt.annotate(str(cm[x][y]), xy=(y,x), ha='center', va='center', color='white',fontsize=12)
            else:
                plt.annotate(str(cm[x][y]), xy=(y,x), ha='center', va='center', color='black',fontsize=12)
                
    # set the title and x,y labels    
    ax.set_xlabel('predicted class',fontsize=12)
    ax.set_ylabel('actual class',fontsize=12)
    ax.set_title(title,y=1.1)

    # to add a colorbar, uncomment
    #fig.colorbar(cax)
    # to save the figure, uncoment
    #plt.savefig(str(title)+'.png',bbox_inches='tight',dpi=250)
    plt.show()