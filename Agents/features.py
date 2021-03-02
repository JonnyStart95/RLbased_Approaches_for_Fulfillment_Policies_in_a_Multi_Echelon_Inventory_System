"""
    This script implements feature transformations. Especially, this is the only script that accesses state dictionary.
"""

import numpy as np

feature_dims = dict()
feature_transformations = dict()

def register(name, func, dim):
    feature_dims[name] = dim
    feature_transformations[name] = func

def state_to_schedule_receipt(state, horizon):
    n = len(state['q_i'])
    schedule_receipt = np.zeros(shape=(n,))
    for i in range(n):
        timelist = list(state['q_p'][i].keys())
        timelist.sort()
        for t in timelist:
            if t > horizon:
                break
            schedule_receipt[i] += state['q_p'][i][t]
    return schedule_receipt
register('schedule_receipt',state_to_schedule_receipt,lambda x:x)

def state_to_inventory_level(state, horizon):
    on_hold = state['q_i']
    schedule_receipt = state_to_schedule_receipt(state, horizon)
    backorder = state['q_d'][1:]
    return on_hold + schedule_receipt - backorder
register('inventory_level',state_to_inventory_level,lambda x:x)

def state_to_flattened_vector(state):
    return np.concatenate([state['q_i'],state['q_ttp'],state['q_d']], axis=0).astype(float)
register('flattened_vector',state_to_flattened_vector,lambda x:3*x+1)

def state_to_beer_game_state(state, horizon):
    return np.concatenate([state_to_inventory_level(state, horizon), state_to_flattened_vector(state)], axis=0).astype(float)
feature_transformations['beer_game_state'] = state_to_beer_game_state
register('beer_game_state',state_to_beer_game_state,lambda x:4*x+1)

def state_to_demand_agent_state(state):
    print("state_to_demand_agent_state:")
    c = np.diff(state['q_d']) - state['q_i'] - state['q_ttp']
    s = np.clip(c,a_min=0,a_max=None)
    d = np.clip(c,a_min=0,a_max=None)

    print("state['q_d']: ", state['q_d'])
    print("np.diff(state['q_d']): ", np.diff(state['q_d']))
    print("c,s,d: ",c,s,d)

    while True:
        print("True")
        c[:-1] += d[1:]
        print("c[:-1]: ", c[:-1])
        print("d[1:]: ", d[1:])
        x = np.clip(c,a_min=0,a_max=None)
        print("x, c, s, d: ", x, c, s, d)
        if (s>=x).all():
            return s
        s,d = x,x-s
register('demand_agent_state',state_to_demand_agent_state,lambda x:x)
