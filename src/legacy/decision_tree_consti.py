#import numpy as np
import copy
#from src.utils import *
#from src.utils_consti import *
from pprint import pprint


#Convention:
#NODE -  Dictionary: {'attribute' : Int, 'value' : double, 'lbranch' : Dictionary, 'rbranch' : Dicstionary}
#xn - Int: Number of training samples
#xm - Int: Number of attributes



#Instance fields:
#tree              Dictionary:  containing the tree that we build
#tree_pruned       #maybe a second tree -> pruned version
#depth             #Int: The depth of the tree
#x_train           #np.array: array holding all the training attributes/signals - dimension (xn,xm)
#xn                #int : number of samples in our training data
#xm                #int: number of signals in our training data
#y_train           #np.array: array holding our training labels - dimension (xn,1)
#x_test            #np.array: array holding all the test attributes/signals - dimension (2000 - xn,xm)
#y_test            #np.array: array holding our training labels - dimension (2000- xn,1)
#???               #Additional fields for Validation



#Class that contains a decision tree. This can be called with training data to create a decision tree
class DecisionTree():
      
    
    #Constructor to create one instance of a decision tree
    def __init__(self,data): 
        self.xn = data.shape[0]
        self.xm = data.shape[1] - 1
        self.x_train = data[:,:-1]
        self.y_train = data[:,-1]
        self.tree,self.depth = self.buildTree(data)
        self.tree_pruned = None
        self.data = data
        
        
        
     
    ##############################################################################
    #Recursivly builds tree from current data partition
    
    #Inputs:
    #data_index: index of the data we are currently looking at
    #
    #Returns: 
    # (NODE,Int) - NODE carrying the node information, Int carrying the depth of this tree
    
    ##############################################################################

    #if all samples in Y_dataset the same label:
    #
    #    return (NODE(attribute, value, null, null) , 0)
    #
    #else 
    #    split = findsplit() #Split of type (attribut,value)
    #    X_dataset_left, X_dataset_right, Y_dataset_left, Y_dataset_right = paritionData(split)
    #    (left,depth_left) = buildTree(dataset_left)
    #    (right,depth_right) = buildtree(dataset_right)
    #    return (NODE,max(depth_left,depth_right))
    
    ##############################################################################
    
    def buildTree(self,data, depth=0):
        labels = np.unique(data[:,-1])
        if len(labels) == 1:
            return ({'attribute' : 'leaf', 'value' : labels[0], 'lbranch' : None, 'rbranch' : None}, depth)
        else:
            split = self.findSplit(data)
            data_left, data_right = self.partitionData(data, split)
            l_branch, l_depth = self.buildTree(data_left, depth+1)
            r_branch, r_depth = self.buildTree(data_right, depth+1)
            return ({'attribute' : split[0], 'value' : split[1], 'lbranch' : l_branch, 'rbranch' : r_branch}, max(l_depth, r_depth))
    
    
    
    ##############################################################################
    #Finds an optimal partition for the current data
    
    #Inputs:
    #data_index: index of the data we are currently trying to find a split
    #
    #Returns:
    #(Int, Double) - Int carrying the attribute we split on, Double carrying the value where to split
    
    ##############################################################################
    
    # compute H(all)
    # maxGain = 0
    # for all attributes:
    #
    #     sort by this attribute
    #          for all different split points:
    #
    #                compute Remainder
    #                if remainder > maxGain: update maxGain save (attribute,value)
    # return (attribute,value)
    
    ##############################################################################
    def findSplit(self, data):
        max_gain = 0
        split_attr = None
        plit_val = None
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
        return (split_attr, split_val)
    
    
    
    ##############################################################################
    #Returns the index of the values belonging to the left branch and right branch
    
    #Inputs:
    #currentIndex: the index of the labels we want to split up
    #split: return value from findSplit (Int, Double)
    
    #Returns: 
    #Dictionary -  {'leftindex' : np.array, 'rightindex' : np.array}
    
    ##############################################################################
    def partitionData(self,data,split):
    
        attribute = split[0]
        split_val = split[1]
        
        data_left = data[data[:,attribute] <= split_val]
        data_right = data[data[:,attribute] > split_val]
        
        return (data_left, data_right)
        
    
    
    
    ##############################################################################
    #Visualizes the tree
    
    #WRITE TESTS FOR ONESIDED TREE!!!
    
    
    ##############################################################################
    def visualize(self):
         return 0   
       
    
    
    
    ##############################################################################
    #Given test data evaluates the performace of our tree:
    #Prints: Confusion Matrx, Recall & Precision per Class, F1-measure, Classification rate
    
    #Inputs: 
    #analog to training data
    
    #Returns: 
    #The values above save in a dictionary? TBD
    
    ##############################################################################
    
    #Initialize confusion maxtrx
    #for each datapoint in testset
    #    run predict
    #    fill confusion matrx
    #
    # Compute all the statistics that we want
    
    ##############################################################################
    def evaluate(self,test_db, pruned_tree = False):
        
        confusion_matrix = np.zeros([4,4])
        
        for point in range(test_db.shape[0]):
            test_label = int(test_db[point][-1])
            predicted_label = int(self.predict(test_db[point][:-1], pruned_tree))
            confusion_matrix[test_label-1,predicted_label-1] += 1
        
        accuracy = np.diag(confusion_matrix).sum()/confusion_matrix.sum()
        precision = np.diag(confusion_matrix)/confusion_matrix.sum(axis=0)
        recall = np.diag(confusion_matrix)/confusion_matrix.sum(axis=1)
        f_1 = 2 * precision * recall / (precision + recall)
        self.stats = {'confusion_matrix' : confusion_matrix,
                      'accuracy' : accuracy,
                      'precision' : precision,
                      'recall' : recall,
                      'f_1' : f_1
        }
        
        return self.stats
 
    
        
    ##############################################################################
    #Given a single point predicts the value of this point with our decision tree
    #Walks through the nested dict until we are at a leaf stating the label of this point
    
    #Inputs:
    #x_point: np.array containg the signal strengths for this point - dimensions 1 x xm
    
    #Returns:
    #Int: The predicted label of this data point
    
    
    ##############################################################################
    def predict(self, signal, pruned_tree = False, tree = None):
        if pruned_tree == True:
            tree = self.tree_pruned
        elif not tree:
            tree = self.tree
        
        def __predict(signal, tree):
            
            if (tree['attribute'] != 'leaf'):
                if (signal[tree['attribute']] <= tree['value']):
                    return __predict(signal, tree['lbranch'])
                else:
                    return __predict(signal, tree['rbranch'])
            else:
                return tree['value']

        return __predict(signal, tree)
    
    
    ##############################################################################
    #We take our current tree and prune it creating a new pruned tree
    #Deepcopy the original first and then do pruning algorithm
    #Inputs:
    
    #Returns: 
    # (NODE,Int) - NODE carrying the node information of the pruned tree, Int carrying the depth of this tree
    
    
    ##############################################################################
    def prune(self, train_db, test_db):
        #tree = self.tree
        #self.tree_pruned = copy.deepcopy(self.tree)
        
        def __prune(tree_pruned, train_db, test_db):
            
            if tree_pruned['attribute'] == 'leaf':
                return 'tree has only one attribute'
        
            if tree_pruned['lbranch']['attribute'] == 'leaf' and tree_pruned['rbranch']['attribute'] == 'leaf':
                #self.tree_pruned = copy.deepcopy(self.tree)
                #calculate accuracy of unpruned tree
                before_accuracy = self.evaluate(test_db, pruned_tree = True)['accuracy']
                #delete last branch
                
                temp_branchl = tree_pruned['lbranch']
                temp_attribute = tree_pruned['attribute']
                temp_value = tree_pruned['value']
                temp_branchr = tree_pruned['rbranch']
                
                
                tree_pruned['attribute'] = 'leaf'
                tree_pruned['lbranch'] = None
                tree_pruned['rbranch'] = None
                label, counts = np.unique(train_db[:,-1], return_counts=True)
                #print(train_db)
                #print('counts: {}'.format(counts))
                tree_pruned['value'] = label[np.argmax(counts)]
                #print('new_value: {}'.format(tree_pruned['value']))
                #calculate new accuracy
                new_accuracy = self.evaluate(test_db, pruned_tree = True)['accuracy']
                #print('before_accuracy = {}'.format(before_accuracy))
                #print('new_accuracy = {}'.format(new_accuracy))
                if new_accuracy >= before_accuracy:
                    #print('HHEEEEEYYYYY')
                    self.tree = copy.deepcopy(self.tree_pruned)
                    #pprint(self.tree)
                    return 1
                else:
                    tree_pruned['attribute'] = temp_attribute
                    tree_pruned['lbranch'] = temp_branchl
                    tree_pruned['rbranch'] = temp_branchr
                    tree_pruned['value'] = temp_value
                    return 2
                    
            else:
                if tree_pruned['lbranch']['attribute'] == 'leaf' and tree_pruned['rbranch']['attribute'] != 'leaf':
                    depthl = 0
                    depthr = __prune(tree_pruned['rbranch'], train_db[train_db[:,tree_pruned['attribute']]>tree_pruned['value']], test_db)
                elif tree_pruned['lbranch']['attribute'] != 'leaf' and tree_pruned['rbranch']['attribute'] == 'leaf':
                    depthr = 0
                    depthl = __prune(tree_pruned['lbranch'], train_db[train_db[:,tree_pruned['attribute']]<=tree_pruned['value']], test_db)
                else:
                    depthr = __prune(tree_pruned['rbranch'], train_db[train_db[:,tree_pruned['attribute']]>tree_pruned['value']], test_db)
                    depthl = __prune(tree_pruned['lbranch'], train_db[train_db[:,tree_pruned['attribute']]<=tree_pruned['value']], test_db)
                return max(depthl,depthr) + 1
        
        current_accuracy = self.evaluate(test_db)['accuracy']
        #print('current_accuracy = {}'.format(current_accuracy))
        while True:
            self.tree_pruned = copy.deepcopy(self.tree)
            self. depth = __prune(self.tree_pruned, train_db, test_db)
            #pprint(self.tree)
            next_accuracy = self.evaluate(test_db)['accuracy']
            #print('next_accuracy = {}'.format(next_accuracy))
            if next_accuracy <= current_accuracy:
                break
            else:
                current_accuracy = next_accuracy       
                
        return self.evaluate(test_db, pruned_tree = True)
    
    
    
    
    
    