import numpy as np
import os
import importlib
from collections import OrderedDict
import copy

from SupplyChain import RandomGenerators as RG
from SupplyChain import CostFunctions as CF

params = {
    'n':3,
    'price':1000.,
    'decision_interval':1,
    'maximal_action':1,
    'demand':RG.ErlangGenerator(a=1.,scale=0.5),
    'leadtimes':[
        RG.DiscreteUniformGenerator(),
        RG.DiscreteUniformGenerator(),
        RG.DiscreteUniformGenerator()
    ],
    'supply':RG.InfiniteGenerator(sign="+"),
    'costs':{
        'ordering':[
            CF.BinaryCost(scale=80.),
            CF.BinaryCost(scale=80.),
            CF.BinaryCost(scale=80.)
        ],
        'holding':[
            CF.LinearCost(unit_cost=3.),
            CF.LinearCost(unit_cost=5.),
            CF.LinearCost(unit_cost=10.)
        ],
        'transportation':[
            CF.LinearCost(unit_cost=3.),
            CF.LinearCost(unit_cost=5.),
            CF.LinearCost(unit_cost=10.)
        ],
        'delaying': CF.LinearCost(unit_cost=50.)
    }
}
class SequentialSupplyChain:

    def __init__(self, params=params):

        self.n = params['n']
        self.price = params['price']
        self.decision_interval = params['decision_interval']
        self.maximal_action = params['maximal_action']
        self.demand = params['demand']
        self.leadtimes = params['leadtimes']
        self.supply = params['supply']
        self.costs = params['costs']

        self.time_step = 0
        pass

    def __get_state__(self):
        return {
            'q_i':np.copy(self.q_i),
            'q_ttp':np.copy(self.q_ttp),
            'q_d':np.copy(self.q_d),
            'q_p':copy.deepcopy(self.q_p)
        }

    def reset(self):

        self.q_i = np.zeros(shape = (self.n,), dtype = int)
        self.q_ttp = np.zeros(shape = (self.n,), dtype = int)
        self.q_d = np.zeros(shape = (self.n+1,), dtype = int)
        self.q_p = [dict() for i in range(self.n)]

        self.time_step = 0

        return self.__get_state__()

    def __integerise__(self, x):
        x = np.array([x])
        y = x.astype(int)
        z = x - y
        return (y + np.random.binomial(1,z))[0]

    def __check_input__(self, action):
        if not isinstance(action, np.ndarray):
            raise TypeError("Action should be a NumPy ndarray.")
        if not action.shape == (self.n,):
            raise ValueError("Action should be of shape {should_be} but {now_be} is found.".format(should_be=(self.n,),
                                                                                                    now_be=action.shape))
        if action.min() < 0:
            raise ValueError("Action should be positive but {} is found. ".format(action.min()))

    def step(self, action):

        self.__check_input__(action)
        action = self.__integerise__(action).astype(int)
        # print("action of sequential:", action)

        reward = 0.
        info = { 'P':0., 'c_o':0., 'c_i':0., 'c_p':0., 'c_d':0. }

        for t in range(self.decision_interval):
            # Costs Update
            # print("\n#Costs Update")
            ordering_cost = np.sum(self.costs['ordering'][i].of(action[i]) for i in range(self.n))# If you take the order, there will be a binary cost
            holding_cost = np.sum(self.costs['holding'][i].of(self.q_i[i]) for i in range(self.n)) #Caculate how much it will take based on the inventory at the time
            transportation_cost = np.sum(self.costs['transportation'][i].of(self.q_ttp[i]) for i in range(self.n))# The product size that need to be transported
            delaying_cost = self.costs['delaying'].of(self.q_d[-1])# The last one element in the q_d will be the cumulated to the backorders
            reward -= ordering_cost + holding_cost + transportation_cost + delaying_cost
            # print("t:", t+1)
            # print('qi:', self.q_i, 'qttp:',self.q_ttp, 'qd:',self.q_d, 'qp:',self.q_p)

            info['c_o'] += ordering_cost 
            info['c_i'] += holding_cost
            info['c_p'] += transportation_cost
            info['c_d'] += delaying_cost
            # print('info:',info)
            # info dictionary will record the cummulation of 4 kinds of costs
            # Order Update
            # print("\n#Order Update")
            self.q_d[:self.n] += action # Take actions namely placing the orders, so the q_d is the orders from upstream
            # print("self.q_d", self.q_d)

            # Production Update
            # print("\n#Production Update")
            for i in range(self.n):
                # print("echelon: ",i+1)
                old_q_p_i = self.q_p[i]
                # print('old_q_p_i:', old_q_p_i)
                # print('q_p:', self.q_p)
                if 1 in old_q_p_i:
                    self.q_i[i] += old_q_p_i[1] # old_q_p_i[1] is the product received at time step
                    self.q_ttp[i] -= old_q_p_i[1]
                new_q_p_i = dict()
                for k in old_q_p_i:
                    if k != 1:
                        new_q_p_i[k-1] = old_q_p_i[k]
                        # print('new_q_p_i:', new_q_p_i)
                self.q_p[i] = new_q_p_i

            # Supply Update
            # print("\n# Supply Update")
            sent = self.__integerise__(min(self.supply.generate(), self.q_d[0]))
            # print("sent", sent)
            new_leadtime = None
            if sent != 0:
                new_leadtime = self.__integerise__(self.leadtimes[0].generate())
                # print("new_leadtime: ", new_leadtime)
                self.q_d[0] -= sent
                if new_leadtime not in self.q_p[0]:
                    self.q_p[0][new_leadtime] = sent
                else:
                    self.q_p[0][new_leadtime] += sent
                self.q_ttp[0] += sent
            # print("self.q_p: ", self.q_p)
            # print("self.q_ttp: ", self.q_ttp)

            # Send Update
            # print("\n#Send Update")
            for i in range(1,self.n):
                # print("i: ", i+1)
                sent = min(self.q_i[i-1], self.q_d[i])
                # print("sent", sent)
                new_leadtime = None
                if sent != 0:
                    new_leadtime = self.__integerise__(self.leadtimes[i].generate())
                    # print("new_leadtime: ", new_leadtime)
                    self.q_d[i] -= sent
                    self.q_i[i-1] -= sent
                    if new_leadtime not in self.q_p[i]:
                        self.q_p[i][new_leadtime] = sent
                    else:
                        self.q_p[i][new_leadtime] += sent
                    self.q_ttp[i] += sent
                # print("self.q_p: ", self.q_p)
                # print("self.q_ttp: ", self.q_ttp)

            # Client Demand Update\
            # print("\n#Client Demand Update")
            self.time_step += 1
            new_demand = self.__integerise__(self.demand.generate())
            self.q_d[-1] += new_demand
            # print("\nnew_demand",new_demand,"\nself.q_d: ", self.q_d)

            # Trade Update
            # print("\n#Trade Update")
            amount = min(self.q_i[-1],self.q_d[-1])
            gain = self.price * amount
            reward += gain
            info['P'] += gain
            self.q_i[-1] -= amount
            self.q_d[-1] -= amount
            # print("amount", amount, "\nself.q_i: ", self.q_i, "\nself.q_d: ", self.q_d)

            action = np.zeros(shape = (self.n,), dtype = int)

            # print("\n-----------------")
        return self.__get_state__(), reward, False, info

    def random_action(self):
        return np.random.randint(low=self.maximal_action+1,size=(self.n,))

    def print_state(self):
        row_format = "{NodeID:8}|{q_i:5}|{q_ttp:7}|{q_d:5}|{q_p}\n"
        print("Sequential Supply Chain Model State:\n",
            row_format.format(NodeID='NodeID',q_i='q_i',q_ttp='q_ttp',q_d='q_d',q_p='q_p'),
            *[row_format.format(NodeID=i+1,q_i=self.q_i[i],q_ttp=self.q_ttp[i],q_d=self.q_d[i],q_p=self.q_p[i])
                                            for i in range(self.n)], "Total Demand: {}\n".format(self.q_d[-1]))