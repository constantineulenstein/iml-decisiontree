import matplotlib as mpl
import copy

from src.utils import *

#DecisionTree class used to build, evaluate, visualize and prune a decision tree
class DecisionTree():
      
    
    #Constructor to create one instance of a decision tree
    def __init__(self,data = None):
        
        self.data = None
        
        if data is not None:
            
            self.xn = data.shape[0]
            self.xm = data.shape[1] - 1
            self.data = data
            self.uniq_labels = len(np.unique(data[:,-1]))
            self.tree,self.depth = self.buildTree(data)
            
     
    ##############################################################################
    #Recursivly builds tree from current data partition
    
    #Inputs:
    #data: data to build the tree
    #
    #Returns: 
    # (NODE,Int) - NODE carrying the node information, Int carrying the depth of this tree
    
    ##############################################################################
    
    def buildTree(self,data):
        
        #Call constructor if tree constructed without data
        if self.data is None:
            self.__init__(data)
        
        labels, count = np.unique(data[:,-1],return_counts = True)

        #If leaf
        if (len(labels) == 1):
            return ({'attribute' : 'leaf','val' : labels[0],'lbranch' : None, 'rbranch' : None},1)
        #If not leaf
        else:
            split = self.findSplit(data)
            branch_left,branch_right = self.partitionData(data,split)
            left,depth_left = self.buildTree(branch_left)
            right,depth_right = self.buildTree(branch_right)
            node = {'attribute' : split[0],'val' : split[1], 'lbranch' : left,'rbranch' : right}
            return (node,max(depth_left,depth_right) + 1)
    
    
    ##############################################################################
    #Finds an optimal partition for the current data. We iterate over all potential splitpoints and choose the point according to the highest information gain
    
    #Inputs:
    #data: the partition of the dataset that we intend to split
    #
    #Returns:
    #(Int, Double) - Int carrying the attribute we split on, Double carrying the value where to split
    
    ##############################################################################
    
    def findSplit(self,data):
        
        split_attr = None
        split_val = None
        maxgain = 0
    
        for attribute in range(data.shape[1] - 1):
            
            splitpoints = np.unique(data[:,attribute])
            for index,point in enumerate(splitpoints):
            
                data_left = data[data[:,attribute] < point]
                data_right = data[data[:,attribute] >= point]
                gain = information_gain(data,data_left,data_right)
            
                if gain > maxgain:
                    split_attr = attribute
                    #For noisy data take average of consequite splitpoints. Check if last splitpoint to avoid out of bounds error
                    split_val = point if index + 1 == len(splitpoints) else (point + splitpoints[index + 1]) / 2
                    splitindex = index
                    maxgain = gain
        
        return (split_attr,split_val)
    
    
    ##############################################################################
    #Returns the data batch belonging to each branch
    
    #Inputs:
    #data: data we want to split
    #split: return value from findSplit (Int, Double)
    
    #Returns: 
    #Tuple -  data of left branch & data of right branch
    
    ##############################################################################
    def partitionData(self,data,split):
    
        attr = split[0]
        val = split[1]
        return data[data[:,attr] < val] , data[data[:,attr] >= val]
    
    
    ##############################################################################
    #Visualizes the tree

    ##############################################################################
    def visualize(self):
            
        #Formats output differently if leaf or internal node
        def __formatNode(node):
            return 'leaf val:{}'.format(node['val']) if node['attribute'] == 'leaf' else 'signal:{} val:{}'.format(node['attribute'],node['val'])
        
        #Recursively prints tree
        def printTree_intern(node, level, last=False, sup=[]):
            
            #Simbols for pretty printing    
            right = '├'
            pipe = '|'
            left = '└'
            dash = '─'
            
            
            def update(left, i):
                if i < len(left):
                    left[i] = '   '
                return left

            print( ''.join(reduce(update, sup, ['{}  '.format(pipe)] * level)) + (left if last else right) + '{} '.format(dash) + __formatNode(node))
        
            if node['attribute'] != 'leaf':
                level += 1
                printTree_intern(node['lbranch'], level, sup=sup)
                printTree_intern(node['rbranch'], level, True, [level] + sup)
            
            
        #Call on root and on both branches
        print("Depth of the tree: {}".format(self.depth))
        print("")
        print(__formatNode(self.tree))
        if self.depth == 1:
            return
        printTree_intern(self.tree['lbranch'],0,last = False,sup=[])
        printTree_intern(self.tree['rbranch'],0,last = True,sup=[0])    
    
    
    ##############################################################################
    #Given test data evaluates the performace of our tree:
    #Prints: Confusion Matrx, Recall & Precision per Class, F1-measure, Classification rate
    
    #Inputs: 
    #test_db = test dataset, 
    #prunedtree (optional = False) used for prune function to destinguish if we evaluate current tree or newly pruned tree
    
    #Returns: 
    #All statistics as a Dictionary
    
    ##############################################################################
    
    def evaluate(self,test_db,prunedtree = False):
        
        xtest = test_db[:,:-1]
        ytest = test_db[:,-1]
        uniq_labels = len(np.unique(ytest))
        self.confusionmatrix = np.zeros((max(self.uniq_labels,uniq_labels),max(self.uniq_labels,uniq_labels)))
        
        for i in range(xtest.shape[0]):
            sample = np.array(xtest[i,:])
            trueval = ytest[i]
            guess = self.predict(sample,prunedtree)
            self.confusionmatrix[int(guess)-1,int(trueval)-1] +=1
        
        precision = np.divide(np.diag(self.confusionmatrix),np.sum(self.confusionmatrix,axis = 1),out=np.zeros_like(np.diag(self.confusionmatrix)), where= np.sum(self.confusionmatrix,axis = 1)!=0)
        #recall = np.diag(self.confusionmatrix) / np.sum(self.confusionmatrix,axis = 0)
        recall = np.divide(np.diag(self.confusionmatrix),np.sum(self.confusionmatrix,axis = 0),out = np.zeros_like(np.diag(self.confusionmatrix)), where = np.sum(self.confusionmatrix,axis = 0) != 0)
        F1score = np.divide(2 * precision * recall, (precision + recall),out=np.zeros_like(2 * precision * recall), where= (precision + recall)!=0)
        posClassRate = np.sum(np.diag(self.confusionmatrix)) / xtest.shape[0]
        
        statistics = {'confusionmatrix' : self.confusionmatrix, 
                      'precision' : precision , 
                      'recall' : recall,
                      'F1score' : F1score,
                      'posClassRate' : posClassRate
                     }
        self.statistics = statistics
        
        return self.statistics
    

    ##############################################################################
    #Given a single point predicts the value of this point with our decision tree
    #Walks through the nested dict until we are at a leaf stating the label of this point
    
    #Inputs:
    #x_point: np.array containg the signal strengths for this point - dimensions 1 x xm
    #prunedtree (optional = False) used for prune function to destinguish if we predict xpoint on current tree or newly pruned tree
    
    #Returns:
    #Int: The predicted label of this data point
    
    ##############################################################################
    def predict(self, x_point,prunedtree = False):
        
        def __predict(node,x_point):
            attr = node['attribute']
            val = node['val']
            if(attr != 'leaf'):
                return __predict(node['lbranch'],x_point) if x_point[attr] < val else __predict(node['rbranch'],x_point)
            else:
                return val
    
        if prunedtree:
            return __predict(self.tree_pruned,x_point)
        
        return __predict(self.tree,x_point)
    
    
    ##############################################################################
    #We take our current tree and prune it creating a new pruned tree
    #Deepcopy the original first and then do pruning algorithm
    #Inputs:
    
    #Returns: 
    # (NODE,Int) - NODE carrying the node information of the pruned tree, Int carrying the depth of this tree
    
    ##############################################################################
    def prune(self,train_data,val_data):
               
        self.val_data = val_data
        self.depth = self.pruneNode(self.tree,train_data)     
        self.evaluate(self.val_data)
        
        
        
        #Internal function called on every subtree
    def pruneNode(self,node,train_data):

        attr = node['attribute']
        val = node['val']
        lbranch = node['lbranch']
        rbranch = node['rbranch']

        #No pruning done one leafs
        if attr == 'leaf':
            return 1

        #Call pruneNode on left and right subtree
        depthl = self.pruneNode(lbranch,train_data[train_data[:,attr] < val])
        depthr = self.pruneNode(rbranch,train_data[train_data[:,attr] >= val])

        #Prune only if node obove two leafs
        if (lbranch['attribute'] == 'leaf') & (rbranch['attribute'] == 'leaf'):
        
            #Get score before pruning
            oldscore = self.evaluate(self.val_data,prunedtree = False)
        
        
            labels, counts = np.unique(train_data[:,-1],return_counts = True)
            maxlabel = labels[np.argmax(counts)]
            
            #Prune tree (delete node above two leafs)
            node['attribute'] = 'leaf'
            node['val'] = maxlabel
            node['lbranch'] = None
            node['rbranch'] = None
        
            #Get score after pruning
            newscore = self.evaluate(self.val_data,prunedtree = False)
        
            #Update tree if score better or the same
            if newscore['posClassRate'] >= oldscore['posClassRate']:
                return 1
            #Otherwise restore tree
            else:
                node['attribute'] = attr
                node['val'] = val
                node['lbranch'] = lbranch
                node['rbranch'] = rbranch
        
        
        #if node not above TWO leafs, recursively call itself to analyze subtree, also return current depth
        return max(depthl,depthr) + 1
    
