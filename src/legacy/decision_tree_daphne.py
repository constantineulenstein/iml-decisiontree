import numpy as np
import matplotlib as mpl

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
    
    def __init__(self,data): 
        self.xn = data.shape[0]
        self.xm = data.shape[1] - 1
        self.x_train = data[:,:-1]
        self.y_train = data[:,-1]
        self.tree = self.buildTree(np.arange(self.xn))
        self.tree_pruned = None
    
    def buildTree(self,data):
        
        labels = np.unique(data[:,-1])
        if len(labels) == 1:
            return ({'attribute' : 'leaf', 'val' : labels[0],'lbranch' : None, 'rbranch' : None},1)
        else:
            branch_left, branch_right, split_node = self.find_split(data)
            left, depth_left = self.buildTree(branch_left)
            right, depth_right = self.buildTree(branch_right)
            node = {'attribute' : split_node[0],'val' : split_node[1], 'lbranch' : left,'rbranch' : right}
            
            return (node, max(depth_left,depth_right) + 1)
    
    def find_split(self, train_data):
        x_data = np.delete(train_data, -1, axis=1)

        biggest_gain = 0
        best_split = []
        best_pair = []

        for attr in range(x_data.shape[1]):
            for value in np.unique(data[:,attribute]):
                split = partition_data(train, [attr,value])
                info_gain = information_gain(train_data,split[0],split[1])
                if info_gain > biggest_gain:
                    biggest_gain = info_gain
                    best_split = split
                    best_pair = [attr,value]
    
    return best_split[0], best_split[1], best_pair


    def partitionData(self,data,split):
        attr = split[0]
        val = split[1]
    return data[data[:,attr] < val] , data[data[:,attr] >= val]
    
    
    #predict the label of a point
    
    def predict(self, test_point, node):
            # until we find the leaf 
        attr = node['attribute']
        if attr != 'leaf':
                # the value of the point at column number attr
            if test_point[attr] <= node['val']:
                return predict(test_point, node['lbranch'] )
            if point[attr] > node['val']:
                return predict(test_point, node['rbranch'] )
        return node['val']
    
    #return accuracy, confusion matrix
    
    def evaluate(self, x_test, y_test, tree):
        correct = 0
        confusion_matrix = np.zeros((4,4))
        for idx in range(len(x_test)):
            predicted_label = predict(x_test[idx], tree[0])
            confusion_matrix[int(y_test[idx])-1][int(predicted_label)-1] +=1

            if predicted_label == y_test[idx]:
                correct += 1
        accuracy = correct / len(x_test)
        
        return accuracy, confusion_matrix

    def precisions(self, confusion_matrix):
        precision_per_class = []
        for room in range(4):
            column = confusion_matrix[:,room]
            precision = column[room] / (np.sum(column))
            precision_per_class[room] = precision
        return precision_per_class
    
    def recalls(self, confusion_matrix):
        recall_per_class = {}
        for room in range(4):
            row = confusion_matrix[:,room]
            recall = column[room] / (np.sum(column))
            recall_per_class[room] = recall
        return recall_per_class
    
    def f1_measures(self, avg_precision, avg_recall):
        f1_measures = np.zeros(4)
        for room in range(4):
            measures_per_class[room] = 2*avg_precision[room]*avg_recall[room] / (avg_precision[room] + avg_recall[room])
        return f1_measures

    



    ##############################################################################
    #We take our current tree and prune it creating a new pruned tree
    #Deepcopy the original first and then do pruning algorithm
    #Inputs:
    
    #Returns: 
    # (NODE,Int) - NODE carrying the node information of the pruned tree, Int carrying the depth of this tree
    
    
    ##############################################################################
    def prune(self):
        
        return None
    
    
    
    
    