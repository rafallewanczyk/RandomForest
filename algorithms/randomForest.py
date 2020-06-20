from pandas import DataFrame
import random
import networkx as net
from algorithms.decisionTree import DecisionTree
from collections import Counter


class RandomForest:
    def __init__(self, num_of_trees, threshold, train_data, min_data_size):
        self.num_of_trees = num_of_trees
        self.threshold = threshold

        self.train_data_intervals = []
        if (len(train_data) < min_data_size):
            raise Exception('min_data_cannot be smaller than size of train_data')

        for i in range(num_of_trees):
            size = random.randint(min_data_size, len(train_data) - 1)
            self.train_data_intervals.append(
                DataFrame(random.sample(train_data.values.tolist(), k=size), columns=train_data.columns))

        self.forest = []
        for i in range(num_of_trees):
            self.forest.append(DecisionTree(self.train_data_intervals[i], self.threshold))
            self.forest[i].build_tree(self.train_data_intervals[i])
            # print(
            #     f"tree nr {i}\n{self.train_data_intervals[i]}\n{net.get_edge_attributes(self.forest[i].graph, 'user_data')}")

    def predict(self, data_interval):
        predictions = []
        for i in range(self.num_of_trees):
            predictions.append(self.forest[i].predict(data_interval))

        # print(predictionsh)
        occurence_count = Counter(predictions)
        # print(occurence_count.most_common(1)[0][0])
        return occurence_count.most_common(1)[0][0]

    def validate(self, validation_data):
        record_list = validation_data.values.tolist()
        columns = validation_data.columns.tolist()
        error_ratio = 0
        for row in record_list:
            answer = self.predict(DataFrame([row[1::]], columns=columns[1::]))
            try:
                # print(f"{answer} ?= {row[0]}")
                error_ratio += answer.split(":")[0] != row[0]
            except AttributeError:
                error_ratio += 1
        return error_ratio / len(record_list)

    def print(self):
        for i in range(self.num_of_trees):
            print(
                f"tree number {i}\ntreshold:{self.forest[i].threshold}\ndata : {self.train_data_intervals[i]}\n{self.forest[i].labels}\n")
