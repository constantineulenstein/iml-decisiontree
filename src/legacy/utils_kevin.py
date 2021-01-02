import numpy as np
from functools import reduce

def get_xydata(data):
    return data[:,:-1] , data[:,-1]

def load_testdata(name : str ):
    
    basepath = '../data/testing_data/'
    path = basepath + name
    data = np.loadtxt(path)

    return data


#Stats list of dictionaries with our evaluation scores -> Returns list with only one measure
def getmeasure(stats,measure):
    return [dic[measure] for dic in stats]


def information_gain(data_all,data_left,data_right):
    
    cnt_all = data_all.shape[0]
    cnt_left = data_left.shape[0]
    cnt_right = data_right.shape[0]
    
    remainder = cnt_left/cnt_all * entropy(data_left) + cnt_right / cnt_all * entropy(data_right)
    gain = entropy(data_all) - remainder
    
    return gain

def entropy(data):
    
    cnt = data.shape[0]
    _,counts = np.unique(data[:,-1],return_counts = True)
    return - sum(counts/cnt * np.log2(counts/cnt))


def crossValidation_split(data,folds):
    
    np.random.seed(42)
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    splits = np.array_split(index,folds)
    return splits
    




#Functions below now in decision tree class
"""
def partition_data(data,split):
    attr = split[0]
    val = split[1]
    return data[data[:,attr] < val] , data[data[:,attr] >= val]

def findSplit(data):
        
    split_attr = None
    split_val = None
    maxgain = 0
    
    for attribute in range(data.shape[1] - 1):
            
        for point in np.unique(data[:,attribute]):
            
            data_left = data[data[:,attribute] < point]
            data_right = data[data[:,attribute] >= point]
            gain = information_gain(data,data_left,data_right)
            
            if gain > maxgain:
                split_attr = attribute
                split_val = point
                maxgain = gain
        
    return (split_attr,split_val)

def buildTree(self,data):
        
    labels = np.unique(data[:,-1])
    if (len(labels) == 1):
        return ({'attribute' : 'leaf','val' : labels[0],'lbranch' : None, 'rbranch' : None},1)
    else:
        split = findSplit(data)
        branch_left,branch_right = partition_data(data,split)
        left,depth_left = buildTree(self,branch_left)
        right,depth_right = buildTree(self,branch_right)
        node = {'attribute' : split[0],'val' : split[1], 'lbranch' : left,'rbranch' : right}
        return (node,max(depth_left,depth_right) + 1)
    

    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
right = '├'
pipe = '|'
left = '└'
dash = '─'


#Chose print format for node or leaf
def printNode(node):
    return 'leaf val:{}'.format(node['val']) if node['attribute'] == 'leaf' else 'signal:{} val:{}'.format(node['attribute'],node['val'])

#Prints the tree
def printTree(root):
    print(printNode(root))
    printTree_intern(root['lbranch'],0,last = False,sup=[])
    printTree_intern(root['rbranch'],0,last = True,sup=[0])
    

#Printing internal
def printTree_intern(node, level, last=False, sup=[]):
    def update(left, i):
        if i < len(left):
            left[i] = '   '
        return left

    print( ''.join(reduce(update, sup, ['{}  '.format(pipe)] * level)) + (left if last else right) + '{} '.format(dash) + printNode(node))
    if node['attribute'] != 'leaf':
        level += 1
        printTree_intern(node['lbranch'], level, sup=sup)
        printTree_intern(node['rbranch'], level, True, [level] + sup)
        

#Predict label of a point: in
def predict(tree,x_point):
    
    def __predict(node,x_point):
        attr = node['attribute']
        val = node['val']
        if(attr != 'leaf'):
            return __predict(node['lbranch'],x_point) if x_point[attr] < val else __predict(node['rbranch'],x_point)
        else:
            return val
    
    root = tree[0]
    return __predict(root,x_point)
        
"""