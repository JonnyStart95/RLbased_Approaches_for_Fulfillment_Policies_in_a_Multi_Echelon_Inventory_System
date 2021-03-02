"""
    This script implements classes for cost functions.
"""

class CostFunction:
    pass

class LinearCost( CostFunction ):

    def __init__( self, unit_cost ):
        self.type = 'linear'
        self.unit_cost = unit_cost
    def of( self, quantity ):
        return self.unit_cost * quantity

class AffineCost( CostFunction ):

    def __init__( self, offset, marginal_cost ):
        self.type = 'affine'
        self.offset = offset
        self.marginal_cost = marginal_cost
    def of( self, quantity ):
        return self.offset + self.marginal_cost * quantity

class BinaryCost( CostFunction ):

    def __init__( self, scale ):
        self.type = 'binary'
        self.scale = scale
    def of( self, quantity ):
        return self.scale * (quantity > 0)
