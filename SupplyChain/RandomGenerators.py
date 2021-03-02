"""
This script implements different sources of demands
"""

import numpy as np
from copy import deepcopy
class RandomGenerator:

    def init_stats(self, count=0, mean=0., meansq=0., var=0., running=True):
        self.mean = mean
        self.var = var

        self.__count__ = count
        self.__meansq__ = meansq

        # self.running is a boolean type
        # Determine whether the stats are computed during the runtime or pre-determined
        # True -> stats are computed according to generated numbers
        # False -> stats are known
        self.running = running

    def init_buffer(self):
        self.__bufferptr__ = 0
        self.__bufferlen__ = 1
        self.__buffer__ = self.generate_buffer(1)

    def update_stats(self, slices):
        slices = np.array(slices)
        l = len(slices)
        self.__count__ += l

        new_mean = (self.__count__-l) / self.__count__ * self.mean + l * np.mean(slices)
        new_meansq = (self.__count__-l) / self.__count__ * self.__meansq__ + l * np.mean(slices**2.)
        self.var = new_meansq - self.__meansq__ - (new_mean**2. - self.mean**2.)
        self.mean = new_mean
        self.__meansq__ = new_meansq

    def generate(self):

        result = self.__buffer__[self.__bufferptr__]
        self.__bufferptr__ += 1

        if self.__bufferptr__ == self.__bufferlen__:
            self.__bufferptr__ = 0
            self.__bufferlen__ *= 2
            if self.running:
                self.update_stats(self.__buffer__)
            self.__buffer__ = self.generate_buffer(self.__bufferlen__)

        return result


from scipy.stats import erlang

class ErlangGenerator(RandomGenerator):
    "This class provides a continous erlang distribution (1, 1)"

    def __init__(self, a = 1, loc = 0, scale = 1):
        self.g = erlang(a=a,loc=loc,scale=scale)
        self.init_stats(mean=self.g.mean(), var=self.g.var(), running=False)
        self.init_buffer()
    def generate_buffer(self, n):
        return self.g.rvs(n)

import numpy.random as npr
class DiscreteUniformGenerator(RandomGenerator):

    def __init__(self, low = 1, high = 2):
        self.low = low
        self.high = high
        self.init_stats(mean=(self.low+self.high-1.)/2., var=((self.high-self.low)**2.-1.)/12., running=False)
        self.init_buffer()
    def generate_buffer(self, n):
        return npr.randint(low=self.low,high=self.high,size=n)

class IntermittentGenerator(RandomGenerator):

    def __init__(self, IDI = 4., CV2 = 2, mean_low = 10, mean_high = 50):
        self.p = 1./IDI
        self.CV2 = CV2
        self.mean_low = mean_low
        self.mean_high = mean_high
        self.init_stats()
        self.init_buffer()
    def generate_buffer(self, n):
        bernouilles = npr.binomial(1,self.p,size=n)
        mus = npr.uniform(low=self.mean_low,high=self.mean_high,size=n)
        succ_ps = mus / self.CV2 / (mus+1)**2.
        succ_ns = mus * succ_ps / (1.-succ_ps)
        return bernouilles * (1+npr.negative_binomial(succ_ns,succ_ps))

class InfiniteGenerator(RandomGenerator):

    def __init__(self, sign="+"):
        self.sign = sign
        self.init_stats(mean=float(self.sign+"inf"), running=False)
        self.init_buffer()
    def generate_buffer(self, n):
        return np.zeros(shape=(n,)) + self.mean
