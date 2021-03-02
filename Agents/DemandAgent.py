"""
    This script implements an agent that orders exactly the same quantities at each location as how much it owns to its downstream.
"""

from . import features as F

params = {}

class DemandAgent:

    def __init__(self, env):
        self.decision_interval = env.decision_interval
        if env.demand.running:
            raise Exception('The expected demand is required.')
        self.expected_demand = env.demand.mean
        pass

    def predict(self, state):
        action = F.state_to_demand_agent_state(state)
        print("action: ", action)
        print("action+: ", action + self.expected_demand * self.decision_interval * .9)
        return action + self.expected_demand * self.decision_interval * .9
        pass

    def witness(self, state, action, next_state, reward, done):
        pass

    def train(self):
        pass
