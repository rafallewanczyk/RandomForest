import numpy as np
from pandas import DataFrame

def read_data(data_file):
    data = []
    names = []
    with open(data_file, 'r') as reader:
        names = reader.readline().strip().split(',')

        for line in reader.readlines():
            data.append(np.array(line.strip().split(',')))
    
    return data, names

def convert_to_frame(data, names):
    return DataFrame(data, columns = names)
     