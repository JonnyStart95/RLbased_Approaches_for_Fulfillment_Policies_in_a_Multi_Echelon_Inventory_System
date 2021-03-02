import random
random.seed(0)
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

import tensorflow as tf
# tf.reset_default_graph()
# tf.set_random_seed(0)

from tqdm import tqdm
import pandas as pd
import os

from SupplyChain.SequentialSupplyChain import SequentialSupplyChain

from Agents.RandomAgent import RandomAgent
from Agents.BenchmarkAgent import BenchmarkAgent
from Agents.SMARTAgent import SMARTAgent
from Agents.DDPGAgent import DDPGAgent
from Agents.DemandAgent import DemandAgent
from Agents.DQNAgent import DQNAgent
from Agents.SarsaLamdaAgent import SarsaLamdaAgent
import shutil

NUM_EPISODES = 100
LEN_EPISODE  = lambda x: 200
DECISION_INTERVAL = 1

#AGENT_CLASS = RandomAgent
#AGENT_CLASS = BenchmarkAgent
# AGENT_CLASS = DemandAgent
# AGENT_CLASS = SMARTAgent

#AGENT_CLASS = DQNAgent
#AGENT_CLASS = DemandAgent
# AGENT_CLASS = SMARTAgent
AGENT_CLASS = SarsaLamdaAgent
#AGENT_CLASS = DQNAgent
#AGENT_CLASS = DDPGAgent

SAVE_DIR = 'tmp/model.ckpt'

def train(agent = None, env = None, params = None,
        num_episodes = NUM_EPISODES,len_episode = LEN_EPISODE,
        decision_interval = DECISION_INTERVAL,
        agent_class = AGENT_CLASS):

    env = SequentialSupplyChain()

    if agent is None:
        if params is None:
            agent = agent_class(env)
        else:
            agent = agent_class(env, params)

    ag = BenchmarkAgent(env)
    reward_history = list()
    td_error_history=list()
    actions_list={}
    for id_episode in tqdm(range(num_episodes)):

        print("id_episode", id_episode)

        cum_reward = 0
        cum_info = {'c_i':0, 'c_p':0, 'c_d':0, 'c_o':0, 'P':0}
        state = env.reset()
        abs_sug = np.zeros(shape=(env.n,))

        len_epi = len_episode(id_episode)
        for id_step in range(len_epi):

            # print("  {:.2f}% of an episode".format(id_step/len_epi*100.), end = '\r')
            action = agent.predict(state)
            actag = ag.predict(state) # actag is the action of the Benchmark

            abs_sug += np.abs(actag-action)
            #print('agent pred: {}, actag pred: {}'.format(action,actag))
            #print(action)
            # env.print_state()

            next_state, reward, done, info = env.step(action)

            done = id_step + 1 == len_epi # done is used to detect if it is the last round of episode

            agent.witness(state, action, next_state, reward, done) # Receive the reward and the next state from the environment
            error=agent.train() # The process for the agent to train itself within this episode

            state = next_state
            cum_reward += reward
            td_error_history.append(error)
            for k in cum_info:
                cum_info[k] += info[k]

        print('{:>5d} epi, ag_r: {:>10.3f}, ag_info: [{:>10.2f}, {:>3.2f}, {:>3.2f}, {:>3.2f}, {:>3.2f}], ag_df:{}'.format(
            id_episode+1, cum_reward/len_epi, *[cum_info[k]/len_epi for k in cum_info.keys()], abs_sug/len_epi), end='\n')

        reward_history.append(cum_reward)
    # finally:
    return reward_history,td_error_history

if __name__ == '__main__':
    reward_history,td_error_history=train()
    
sarsa_lamda_reward=reward_history
sarsa_lamda_td_error=td_error_history

benchmark_reward=reward_history
benchmark_td_error=td_error_history




x = np.linspace(1,100,100)  
plt.plot(x,sarsa_lamda_reward, 'black',label="SarsaLamda_lamda")
plt.plot(x,benchmark_reward,'silver',label="BenchMark")
plt.xlabel('Iterarions')
plt.ylabel('Cumulated Reward')
plt.legend(loc=0,ncol=1)
plt.show()
y = np.linspace(1, 20000, 20000)  
plt.plot(y,sarsa_lamda_td_error , 'black',label='SarsaLamda_lamda')
plt.plot(y,benchmark_td_error , 'silver',label="BenchMark")
plt.xlabel('Iterarions')
plt.ylabel('td error')
plt.legend(loc=0,ncol=1)
plt.show()


