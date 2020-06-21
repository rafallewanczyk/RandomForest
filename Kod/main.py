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
    parser = argparse.ArgumentParser(description="ID3 algorithm")
    parser.add_argument('-d', type=str, help="Data file to import")
    parser.add_argument('-k', type=int, help='k-fold cross validation')

    args = parser.parse_args()

    if args.d is None or args.k is None:
        print('Missing input file name or k')
        return

    data, names = read_data(args.d)
    random.shuffle(data)

    for j in [0.1, 0.4]:
        min_data = int(len(data) / 2)
        results_roulette = []
        results_standard = []

        for i in range(1, 30):
            num_of_trees = i
            threshold = j
            roulette_errors, roulette_avg = cross_validation(data, names, args.k, num_of_trees, threshold
                                                             , min_data, 'roulette')
            results_roulette.append(roulette_avg)

            if threshold == 0.1:
                normal_errors, normal_avg = cross_validation(data, names, args.k, num_of_trees, threshold
                                                             , min_data, 'standard')
                results_standard.append(normal_avg)

        print(f"threshold = {j}, min_data = {min_data}, roulette")
        print(results_roulette)
        print(f"threshold = {j}, min_data = {min_data}, standard")
        print(results_standard)


if __name__ == '__main__':
    main()
