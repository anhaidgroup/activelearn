
from array import array
import random
from bisect import bisect


class WeightedRandomSampler(object):
    def __init__(self, weights):
        self.totals = array('d')
        self.sum_of_weights = 0.0
        running_total = 0
        for weight in weights:
            running_total += weight
            self.totals.append(running_total)
        self.sum_of_weights = running_total

    def next(self):
        rnd = random.random() * self.sum_of_weights
        snum = bisect(self.totals, rnd)
        return min(snum, len(self.totals)-1)
