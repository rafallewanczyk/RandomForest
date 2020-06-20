from .decisionTree import DecisionTree
from random import random


class RouletteDecisionTree(DecisionTree):

    def best_attribute(self, data_interval, entropy):
        information_gains = self.information_gains(data_interval, entropy)
        gains_sum = sum(information_gains.values())
        if gains_sum == 0: return ''
        gains_probabilities = []

        for parameter in information_gains.keys():
            odds = information_gains[parameter] / gains_sum
            gains_probabilities.append((parameter, odds))

        lottery = random()
        temp_sum = 0
        for parameter in gains_probabilities:
            temp_sum += parameter[1]
            if lottery < temp_sum:
                return parameter[0]
