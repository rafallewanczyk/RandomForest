from .decisionTree import DecisionTree
from random import random


class RouletteDecisionTree(DecisionTree):

    def best_attribute(self, data_interval, entropy):
        information_gains = self.information_gains(data_interval, entropy)

        gains_probabilities = self.calculate_probabilities(information_gains)
        if len(gains_probabilities) == 0: return ''

        for probability in gains_probabilities:
            if probability[1] < self.threshold:
                del information_gains[probability[0]]

        gains_probabilities = self.calculate_probabilities(information_gains)
        if len(gains_probabilities) == 0: return ''

        lottery = random.random()
        temp_sum = 0
        for parameter in gains_probabilities:
            temp_sum += parameter[1]
            if lottery < temp_sum:
                return parameter[0]

    def calculate_probabilities(self, information_gains):
        gains_probabilities = []
        gains_sum = sum(information_gains.values())
        if gains_sum == 0: return []
        for parameter in information_gains.keys():
            odds = information_gains[parameter] / gains_sum
            gains_probabilities.append((parameter, odds))
        return gains_probabilities
