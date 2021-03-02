"""
This script implements a benchmark agent for sequential supply chain model.
"""

import numpy as np

from . import features as F

params = {
    'safety_stock':[2,2,6]
}

class BenchmarkAgent:
    """The Benchmark Agent object.
    Public Methods to interact with:
        predict: safety stock + target stock - inventory_T
                 (target stock: see belows)
                 (inventoty_T: on-hold + upstream order_T- downstream order_T)
                 _T(an abuse of notation, T time later): decision_interval
        witness: there is nothing to witness
        train: there is nothing to train
    Params:
        safety_stock, decision_interval: predefined
        target_stock: day * averge demand of the distribution
                 (day: time needed to be covered + lead time,
                       cover time: if we order according to the EOQ model, how many days can be covered
                 lead time: average lead time of the distribution)
   """
    def __init__(self, env, params=params):
        self.n = env.n
        self.safety_stock = self.__parse_safety_stock__(self.n, params['safety_stock'])
        self.target_stock = self.__compute_target_stock__(env)
        self.decision_interval = env.decision_interval

    def __parse_safety_stock__(self, n, vals):
        # Parse the assigned parameters of the safety stock to a vector in an out-of-this-world-complex way
        # so that you cannot understand (yeah!)"
        vec = np.zeros(shape=(n,), dtype = int)
        if isinstance(vals, int):
            vec[-1] = vals
            return vec
        if isinstance(vals, list):
            all_ints = True
            for e in vals:
                if not isinstance(e, int):
                    all_ints = False
            if all_ints:
                used_dim = min(len(vals), n)
                vec[-used_dim:] = np.array(vals)
                return vec
        raise Exception("safety_stock should be list of integers or int")

    def __compute_target_stock__(self, env):
        if env.demand.running:
            raise Exception('The expected demand of Supply Chain Model is required.')
        d = env.demand.mean

        leadtimes = list()
        for i in range(self.n):
            if env.leadtimes[i].running:
                raise Exception('The expected leadtime of Supply Chain Model is required.')
            leadtimes.append(env.leadtimes[i].mean)
        leadtimes = np.array(leadtimes)

        ordering_costs = list()
        for i in range(self.n):
            if env.costs['ordering'][i].type != 'binary':
                raise Exception('The ordering cost should be binary.')
            ordering_costs.append(env.costs['ordering'][i].scale)
        ordering_costs = np.array(ordering_costs)

        holding_costs = list()
        for i in range(self.n):
            if env.costs['holding'][i].type != 'linear':
                raise Exception('The holding cost should be linear.')
            holding_costs.append(env.costs['holding'][i].unit_cost)
        holding_costs = np.diff([0.]+holding_costs)

        # EOQ: optimal ordering lead time
        optimal_situations = np.sqrt(ordering_costs * 2. / (holding_costs * d))
        evaluate = lambda i,x:(ordering_costs[i]/x+.5*d*holding_costs[i]*x)

        # DP
        dp_dim = int(max(optimal_situations)) + self.n
        dp_array = np.zeros(shape = (self.n+1,dp_dim))
        for i in range(self.n):
            for j in range(dp_dim):
                dp_array[i+1][j] = evaluate(i,j+1) + min(dp_array[i,j:])

        # Track Back
        time_to_cover = np.zeros(shape=(self.n+1,), dtype=int)
        time_to_cover[-1] = 1
        for i in range(self.n,0,-1):
            i = i-1
            time_to_cover[i] = np.argmin(dp_array[i+1][time_to_cover[i+1]-1:]) + time_to_cover[i+1]

        return ((time_to_cover[:self.n] + leadtimes) * d).astype(int)

    def predict(self, state):
        inventory_level = F.state_to_inventory_level(state, self.decision_interval)
        return (self.safety_stock + self.target_stock - inventory_level).clip(min = 0)

    def witness(self, state, action, next_state, reward, done):
        pass

    def train(self):
        pass
