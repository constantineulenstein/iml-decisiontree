import sys
import os
sys.path.append('../')
os.getcwd()

import unittest

#The following functions are tests:

#information_gain(data_all,data_left,data_right)
#entropy(data)
#partition_data(data,split)
#find_split(data)





#Put your files in a utils_<name> file and make sure they are named the same then you can run theses tests on your functions
from src.utils_kevin import *
from src.utils import *
from src.DecisionTree import DecisionTree

class test_DecisionTree(unittest.TestCase):
    
    #Loads Testdata
    def setUp(self):
        self.data = load_testdata('clean_dataset_small.txt')
        self.x_train = self.data[:,:-1]
        self.y_train = self.data[:,-1]
    
    
    def test_setupTree(self):
        decisiontree = DecisionTree(self.data)
        self.assertEqual(decisiontree.xm, self.x_train.shape[1], "Should be " + str(self.x_train.shape[1]) + " attributes")
        self.assertEqual(decisiontree.xn,self.x_train.shape[0],"Should be " + str(self.x_train.shape[0]) + " datapoints")
        np.testing.assert_array_equal(decisiontree.data, self.data, "x_train data does not match")
      
    def test_partition_data(self):
        data = load_testdata('fake_dataset_small.txt')
        split = (2,-15)
        
        left,right = DecisionTree.partitionData(self,data, split)
        np.testing.assert_array_equal(left,data[data[:,split[0]] < split[1]],"Split arrays do not match")
        np.testing.assert_array_equal(right,data[data[:,split[0]] >= split[1]],"Split arrays do not match")
    
    def test_findSplit_easy(self):
        data = load_testdata('fake_dataset_small.txt')
        split = DecisionTree.findSplit(None,data)
        self.assertEqual(split, (2,-15), "Should be (2,-15)")
        left,right = DecisionTree.partitionData(self,data,split)
        self.assertEqual( DecisionTree.findSplit(None,left), (None,None), "Should be (None,None)")
        self.assertEqual( DecisionTree.findSplit(None,right), (4,-20), "Should be (4,-20)")

    def test_evaluate(self):
        data = get_data('clean_dataset.txt')
        data = data[data[:,-1] != 3]
        return None
    
    
    
    

class test_utils(unittest.TestCase):
    
    
    #Loads Testdata
    def setUp(self):
        self.data = load_testdata('clean_dataset_small.txt')
    
    #Tests entropy function
    def test_entropy(self):
        self.assertEqual(entropy(self.data), 1.4591479170272448, "Should be 1.4591479170272448")

    #Tests information gain function
    def test_information_gain(self):
        
        #Basic
        data_left = self.data[0:3,:]
        data_right = self.data[3:,:]
        self.assertEqual(information_gain(self.data,data_left,data_right), 1.0, "Should be 1.0")  
        

if __name__ == '__main__':
    unittest.main()