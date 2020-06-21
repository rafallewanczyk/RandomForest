import networkx as net
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from math import log
import matplotlib.pyplot as plt
from pandas import DataFrame
import random

class DecisionTree():

    def __init__(self, train_data, threshold):
        self.data = train_data
        self.threshold = threshold
        self.graph = net.DiGraph()
        self.counter = 0
        self.labels = {}

    def build_tree(self, data_interval, parent=None, attribute_state=None):
        classes = data_interval['class'].unique().tolist()
        data_len = len(data_interval)
        attribute = (self.best_attribute(
            DataFrame(random.sample(data_interval.values.tolist(), k=random.randint(int(data_len / 3), data_len - 1)),
                      columns=data_interval.columns), self.total_entropy(data_interval)), self.counter)
        self.counter += 1

        if attribute[0] == "":
            s = ""
            max_ratio = 0
            for state in classes:
                counter = 0
                for x in data_interval['class']:
                    if x == state:
                        counter += 1
                ratio = counter / len(data_interval['class'])
                if (ratio > max_ratio):
                    max_ratio = ratio
                    s = (f"{state}")
                    # s = (f"{state}:%.2f" % (counter / len(data_interval['class'])))
            self.graph.add_node((s, self.counter))
            self.labels[(s, self.counter)] = s
            if parent:
                self.graph.add_edge(parent, (s, self.counter), user_data=attribute_state)
            else:
                self.root = (s, self.counter)

            self.counter += 1
            return

        self.graph.add_node(attribute)
        self.labels[attribute] = attribute[0]
        if parent:
            self.graph.add_edge(parent, attribute, user_data=attribute_state)
        else:
            self.root = attribute

        for state in data_interval[attribute[0]].unique():
            self.build_tree(data_interval.loc[data_interval[attribute[0]] == state], attribute, state)

    def total_entropy(self, data_interval):
        entropy = 0
        number_of_choices = len(data_interval['class'].unique())
        if number_of_choices == 1:
            return 0
        for x in data_interval['class'].unique():
            ratio = data_interval.loc[data_interval['class'] == x].shape[0] / data_interval.shape[0]
            entropy += -ratio * log(ratio, number_of_choices)
        return entropy

    def information_gains(self, data_interval, entropy):
        gains = {}
        for c in data_interval.columns:
            if c == 'class': continue
            information = 0
            for state in data_interval[c].unique():
                df = data_interval.loc[data_interval[c] == state]
                information += df.shape[0] / data_interval.shape[0] * self.total_entropy(df)
                gains[c] = entropy - information

        return gains

    def best_attribute(self, data_interval, entropy):
        best_parameter, best_gain = '', 0
        information_gains = self.information_gains(data_interval, entropy)
        for parameter in information_gains.keys():
            if information_gains[parameter] > best_gain:
                best_gain = information_gains[parameter]
                best_parameter = parameter
        return best_parameter

    def predict(self, data_interval, attribute=None):
        if attribute == None: attribute = self.root
        if len(list(self.graph.neighbors(attribute))) == 0:
            return attribute[0]

        for node in list(self.graph.neighbors(attribute)):
            if self.graph.get_edge_data(attribute, node)['user_data'] == data_interval[attribute[0]][0]:
                return self.predict(data_interval, node)

    def validate(self, validation_data):
        record_list = validation_data.values.tolist()
        columns = validation_data.columns.tolist()
        error_ratio = 0
        for row in record_list:
            answer = self.predict(DataFrame([row[:-1]], columns=columns[:-1]))
            try:
                error_ratio += answer.split(":")[0] != row[0]
            except AttributeError:
                error_ratio += 1
        return error_ratio / len(record_list)
