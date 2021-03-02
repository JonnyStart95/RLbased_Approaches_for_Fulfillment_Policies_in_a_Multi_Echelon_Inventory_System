"""
    Proximal Policy Optimisation
"""

params = {

}

class PPOAgent:

    def __init__( self, env, params ):

        self.n = env.n
        self.maximal_action = env.maximal_action

        self.__build_network__()
        pass

    def __build_network__( self ):

        
        pass

    def predict( self, state ):
        pass

    def witness( self, state, action, next_state, reward, done ):
        pass

    def train( self ):
        pass
