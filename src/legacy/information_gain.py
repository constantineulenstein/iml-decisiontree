def information_gain(data, data_X, data_Y):
    remainder = len(data_X)/(len(data_X) + len(data_Y))*entropy(data_X) + len(data_Y)/(len(data_X)+len(data_Y))*entropy(data_Y)
    gain = entropy(data) - remainder
    return gain