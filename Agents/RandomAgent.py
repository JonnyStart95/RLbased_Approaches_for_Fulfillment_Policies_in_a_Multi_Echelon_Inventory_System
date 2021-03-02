"""
This script implements a random agent for the sequential supply chain model.

Parameters:

    There is no parameters.

Specifications:

    There is no specifications.
"""

class RandomAgent:

    def __init__( self, env, params ):
        self.env = env
        self.inspected_values = dict()

    def predict( self, state ):
        return self.env.random_action()

    def witness( self, state, action, next, reward, done ):
        pass

    def train( self ):
        pass
