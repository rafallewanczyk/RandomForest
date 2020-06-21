from .decisionTree import DecisionTree
from .rouletteDecisionTree import RouletteDecisionTree
from .randomForest import RandomForest
from pandas import DataFrame
from statistics import mean

def cross_validation(data, names, k, num_of_trees, threshold, min_data_size, tree):
    objects_number = len(data)
    lines_step = objects_number / k
    validation_start = 0
    validation_stop = lines_step
    error_ratios = []
    tree = None

    for i in range(0, k):
        train = []
        validation = []
        line_number = 1

        for line in data:
            if line_number > validation_start and line_number < validation_stop:
                validation.append(line)
            else:
                train.append(line)
            line_number += 1

        train_frame = DataFrame(train, columns=names)
        validation_frame = DataFrame(validation, columns=names)
        validation_start += lines_step
        validation_stop += lines_step

        # print(f"validation nr{i}")
        forest = RandomForest(num_of_trees, threshold, train_frame, min_data_size, tree)

        error = forest.validate(validation_frame)
        error_ratios.append(error)


        # tree = DecisionTree(train_frame) if n else RouletteDecisionTree(train_frame)
        # tree.build_tree(train_frame)
        # error = tree.validate(validation_frame)
        # error_ratios.append(error)

    return error_ratios, mean(error_ratios)