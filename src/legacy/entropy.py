import numpy as np
def entropy(data):
    '''calculate entropy of a sample set'''
    
    # count element in each category
    labels = data[:,-1]
    _,count_labels = np.unique(labels,return_counts =True)
  
    p_xk = count_labels / count_labels.sum()
    
    H_entropy = sum(-p_xk * np.log2(p_xk))
    
    return H_entropy