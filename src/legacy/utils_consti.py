import numpy as np

def get_data(train_percentage):
    #data = np.loadtxt("../data/wifi_db/clean_dataset.txt")
    data = np.loadtxt("iml_decisiontree/data/wifi_db/clean_dataset.txt")
    np.random.seed(11111)
    #get all the ids of our dataset and select random ids from the dataset for the train_set
    data_ids = np.arange(2000)
    train_ids = np.random.choice(data_ids, np.int_(len(data)*train_percentage), replace=False)
    #get the ids for the test_set
    test_ids = np.delete(np.arange(2000), train_ids)
    #build train and test sets
    train_data = data[train_ids, :]
    test_data = data[test_ids,:]
    return train_data, test_data
train_data, test_data = get_data(0.8)

def entropy(data):
    
    cnt = data.shape[0]
    _,counts = np.unique(data[:,-1],return_counts = True)
    return - sum(counts/cnt * np.log2(counts/cnt))

def information_gain(data_all,data_left,data_right):
    
    cnt_all = data_all.shape[0]
    cnt_left = data_left.shape[0]
    cnt_right = data_right.shape[0]
    
    remainder = cnt_left/cnt_all * entropy(data_left) + cnt_right / cnt_all * entropy(data_right)
    gain = entropy(data_all) - remainder
    
    return gain

def perform_cross_validation(folds, data, pruned = False):
    fold_size = data.shape[0]/folds
    splits = np.array_split(data,folds)
    individual_stats = []
    for fold in range(folds):
        #create test and train_data for each fold
        test_data = splits[fold]
        train_data = np.vstack([data[:int(fold*fold_size)],data[int((fold+1)*fold_size):]])
        tree = DecisionTree(train_data)
        if pruned == True:
            tree.prune(train_data, test_data)
            print('Tree was pruned!')
        print('Built tree {}'.format(fold))
        individual_stats.append(tree.evaluate(test_data))
        print('Evaluated tree {}'.format(fold))
    
    avg_accuracy = sum(individual_stats[i]['accuracy'] for i in range(folds))/folds
    avg_confusion_matrix = sum(individual_stats[i]['confusion_matrix'] for i in range(folds))/folds
    avg_precision = sum(individual_stats[i]['precision'] for i in range(folds))/folds
    avg_recall = sum(individual_stats[i]['recall'] for i in range(folds))/folds
    avg_f1 = sum(individual_stats[i]['f_1'] for i in range(folds))/folds    
    
              
    avg_stats = {'avg_confusion_matrix': avg_confusion_matrix,
             'avg_accuracy' : avg_accuracy,
             'avg_precision' : avg_precision,
             'avg_recall' : avg_recall,
             'avg_f1' : avg_f1            
    }
    
    
    return avg_stats
        
        
        