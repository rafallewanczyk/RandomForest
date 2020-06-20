from algorithms.decisionTree import DecisionTree
from pandas import DataFrame
import numpy as np
import networkx as net
import matplotlib.pyplot as plt
from algorithms.rouletteDecisionTree import RouletteDecisionTree
import argparse
from data.reader import read_data, convert_to_frame
from algorithms.randomForest import RandomForest
from algorithms.cross_validation import cross_validation
import random
def main():
    random.seed(156)
    parser = argparse.ArgumentParser(description="ID3 algorithm")
    parser.add_argument('-d', type=str, help="Data file to import")
    parser.add_argument('-k', type=int, help='k-fold cross validation')

    args = parser.parse_args()

    if args.d is None or args.k is None:
        print('Missing input file name or k')
        return

    data, names = read_data(args.d)
    random.shuffle(data)
    # forest = RandomForest(num_of_trees= 4, threshold= 1, train_data = DataFrame(data, columns = names), min_data_size = 2)
    # record = DataFrame(data, columns = names)[0:3]
    # forest.print()

    #
    # print(f"rekord :{record} ")
    # print(f"error ratio : {forest.validate(record)}")
     # normal_errors, normal_avg = cross_validation(data, names, args.k, True)
    roulette_errors, roulette_avg = cross_validation(data, names, args.k, 12, 0.1,7)
    # print(f'Normal tree === errors ratio: {normal_errors}, avarage: {normal_avg}')
    print(f'Roulette tree === errors ratio: {roulette_errors}, avarage: {roulette_avg}')
    



if __name__ == '__main__':
    main()
