"""
This script implements a original SMART agent for sequential supply chain model.

Parameters:

    learning_rate:
        function receiving time step as input.
        Definition of learning rate schema

    exploration_rate:
        function receiving time step as input.
        Definition of explore rate schema

    encoding:
        (dim, min, width). (int, int, int)
        inventory level will be encoded into dim possible values. The bins corresponding to these values are:
            (-inf,min+width),
            [ min+width, min+width*2),
            [ min+width*2, min+width*3),
            ...,
            [ min+width*(dim-1), +inf)

Specifications:

    There is no specification.
"""

import numpy as np
import random
import sys

from . import features as F

params = {
    'learning_rate':lambda t:.01/(.01*t+1.),
    'exploration_rate':lambda t:.1/(.01*t+1.),
    'encoding':{
        'dim':10,
        'min':-10,
        'width':2
    }
}

class SMARTAgent:

    def __init__(self, env, params=params):

        self.n = env.n
        self.maximal_action = env.maximal_action
        self.random_action = env.random_action
        self.decision_interval = env.decision_interval

        self.encoding = params['encoding']
        self.encoding['max'] = self.encoding['min'] \
                + self.encoding['width'] * self.encoding['dim']
        self.state_dim = (self.encoding['dim'],) * self.n
        self.action_dim = (self.maximal_action+1,) * self.n
        self.dp_array = np.zeros(shape = self.state_dim + self.action_dim)

        self.learning_rate = params['learning_rate']
        self.exploration_rate = params['exploration_rate']

        self.cumulated_reward = 0.
        self.total_time = 0
        self.reward_rate = 0.

        self.inspected_values = dict()
        self.inspected_values['reward_rate'] = list()
        self.inspected_values['td_error'] = list()

    def __encode_inventory_level__(self, inventory_level):

        return tuple(np.floor((
                        inventory_level.clip(
                                min = self.encoding['min'] + .5,
                                max = self.encoding['max'] - .5)
                        - self.encoding['min'])
                / self.encoding['width']).astype(int))

    def predict(self, state):

        encoded = self.__encode_inventory_level__(F.state_to_inventory_level(state, self.decision_interval))
        self.state = encoded

        self.is_explore = np.random.uniform() < self.exploration_rate(self.total_time)

        action_space = self.dp_array[self.state]
        is_best_actions = action_space==np.amax(action_space)

        if self.is_explore:
            cands = list(zip(*np.where(np.logical_not(is_best_actions))))
            if len(cands) == 0:
                self.action = tuple(self.random_action())
            else:
                self.action = random.sample(cands, 1)[0]
        else:

            drawn = np.random.randint(low = np.sum(is_best_actions))
            self.action = tuple(col[drawn] for col in np.where(is_best_actions))

        return np.array(self.action)

    def witness(self, state, action, next_state, reward, done):

        self.next_state = self.__encode_inventory_level__(F.state_to_inventory_level(next_state, self.decision_interval))
        self.reward = reward
        self.done = done

    def train(self):

        learning_rate = self.learning_rate(self.total_time)

        td_error = (self.reward - self.decision_interval * self.reward_rate + np.amax(self.dp_array[self.next_state]))
        self.dp_array[self.state][self.action] = (1-learning_rate) * self.dp_array[self.state][self.action] + learning_rate * td_error

        if not self.is_explore:
            self.cumulated_reward += self.reward
            self.total_time += self.decision_interval
            self.reward_rate = self.cumulated_reward / self.total_time

        self.inspected_values['reward_rate'].append(self.reward_rate)
        self.inspected_values['td_error'].append(td_error)

        if self.done:
            self.cumulated_reward = 0.
            self.total_time = 0
            self.reward_rate = 0.
