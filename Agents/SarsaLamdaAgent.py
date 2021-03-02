#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:06:39 2019

@author: william
"""
import numpy as np
import pandas as pd
import json

params = {
    'learning_rate':0.9,
    'reward_decay':0.9,
    'e_greedy':0.9,
    'trace_decay':0.9,
    'maximal_action':1
}

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)  
    
class SarsaLamdaAgent:
    def __init__(self, env, params=params):
        self.env = env
        self.lr = params['learning_rate']
        self.gamma = params['reward_decay']
        self.epsilon = params['e_greedy']
        self.lambda_ = params['trace_decay']
        self.max_action=params['maximal_action']
        self.actions=[]
        self.actions_list={}
        self.number_actions=0
        self.action_key=0      
        self.next_state={}
        self.state = env.reset()
        for i in range((self.max_action)+1):
            for j in range ((self.max_action)+1):
                for k in range ((self.max_action)+1):
                    self.actions_list[self.number_actions]=np.array([i,j,k])
                    self.actions.append(self.number_actions)
                    self.number_actions+=1
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.eligibility_trace = self.q_table.copy()
    
    def check_state_exist(self, state):
        state=str(state['q_d'])
        if state not in self.q_table.index:
                # append new state to q table
            to_be_append=pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,)
            self.q_table = self.q_table.append(to_be_append)
            # append state value to the states_list
            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)
            
 
    def predict(self, state):
        self.check_state_exist(state)
        state=str(state['q_d'])
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[state, :]
            # some actions may have the same value, randomly choose on in these actions
            self.action_key = np.random.choice(state_action[state_action == np.max(state_action)].index)
            action=self.actions_list[self.action_key]
        else:
            # choose random action
            action = self.env.random_action()
        return action



    def witness( self, state, action, next, reward, done ):
        self.state=state
        self.action=action
        self.reward=reward
        self.next=next # next is the next state which is got from the environment
        self.done=done
    # Witness is used to get the parameters used for train function
    
    def train(self):
        self.check_state_exist(self.state)
        self.state=str(self.state['q_d'])
        q_predict = self.q_table.loc[self.state, self.action_key] # q_predict is the reward you get at the moment
        # Method 2:
        self.eligibility_trace.loc[self.state, :] *= 0
        self.eligibility_trace.loc[self.state, self.action_key] = 1
        self.check_state_exist(self.next)
        self.next=str(self.next['q_d'])
        if self.done != True:
            q_target = self.reward + self.gamma * self.q_table.loc[self.next, self.action_key]  # next state is not terminal
        else:
            q_target = self.reward  # next state is terminal
        error = q_target - q_predict
        
        # increase trace amount for visited state-action pair

        # Method 1:
        # self.eligibility_trace.loc[s, a] += 1


        # Q update
        self.q_table += self.lr * error * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma*self.lambda_     
        return error
        
        