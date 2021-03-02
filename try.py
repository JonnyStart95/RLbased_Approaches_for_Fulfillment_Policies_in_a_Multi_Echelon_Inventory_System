import tensorflow as tf
import numpy as np
import pandas as pd
from SupplyChain import CostFunctions as CF
from SupplyChain import RandomGenerators as RG
import copy
import json
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)

print(C)



# Try by MINGLIANG
supply = RG.InfiniteGenerator(sign="+")
 def __integerise__(x):
        x = np.array([x])
        y = x.astype(int)
        z = x - y
        return (y + np.random.binomial(1,z))[0]
action=[1,2,3] # 0: factory, 1:Wholeseller, 2: Retailer, 3: Market
q_i = [2,2,2]
q_ttp = [2,2,2]
q_d = [2,2,3,5] # The backorders will also be cummulated in the last elements from customers
q_p = [{2:1},{2:3},{2:4}] # Key stands for the lead time, the value stands fot the production for each echelon
time_step=0
price=1000.
leadtimes=[
        RG.DiscreteUniformGenerator(),
        RG.DiscreteUniformGenerator(),
        RG.DiscreteUniformGenerator()
    ]
demand=RG.ErlangGenerator(a=1.,scale=1.)

    def __get_state__():
        return {
            'q_i':np.copy(q_i),
            'q_ttp':np.copy(q_ttp),
            'q_d':np.copy(q_d),
            'q_p':copy.deepcopy(q_p)
        }
        
state=__get_state__()       

print(q_i)
costs={
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
ordering_cost = np.sum(costs['ordering'][i].of(action[i]) for i in range(3))# If you take the order, there will be a binary cost
holding_cost = np.sum(costs['holding'][i].of(q_i[i]) for i in range(3)) #Caculate how much it will take based on the inventory at the time
transportation_cost = np.sum(costs['transportation'][i].of(q_ttp[i]) for i in range(3))# The product size that need to be transported
delaying_cost = costs['delaying'].of(q_d[-1])# The last one element in the q_d will be the cumulated to the backorders

reward = ordering_cost + holding_cost + transportation_cost + delaying_cost

info = { 'P':0., 'c_o':0., 'c_i':0., 'c_p':0., 'c_d':0. }

info['c_o'] += ordering_cost 
info['c_i'] += holding_cost
info['c_p'] += transportation_cost
info['c_d'] += delaying_cost
          # Order Update
            # print("\n#Order Update")
q_d[:3] += action
 # Production Update
            for i in range(3):
                # print("echelon: ",i+1)
                old_q_p_i = q_p[i] 

                if 1 in old_q_p_i:# If the key value is equal to 1, it means that the products as been delivered
                    q_i[i] += old_q_p_i[1] # So the products will be put into the inventory 
                    q_ttp[i] -= old_q_p_i[1] # The products being transported will be updated
                new_q_p_i = dict() 
                for k in old_q_p_i:
                    if k != 1:
                        new_q_p_i[k-1] = old_q_p_i[k]
                        # print('new_q_p_i:', new_q_p_i)
                q_p[i] = new_q_p_i

            # Supply Update
            sent = __integerise__(min(supply.generate(), q_d[0])) # The first element in q_d is the demand generated to represent the demand from customers
            # To get the minimum between the generated number and the demand

new_leadtime = None
            if sent != 0:
                new_leadtime = __integerise__(leadtimes[0].generate()) # Use the generator to generate a random leadtime
                # print("new_leadtime: ", new_leadtime)
                q_d[0] -= sent
                if new_leadtime not in q_p[0]: 
                    q_p[0][new_leadtime] = sent # q_p[0] is the production status in the factory, there will be different batches of products recorded in the dictionary
                else:
                    q_p[0][new_leadtime] += sent # If different batches of products have the same leadtime, the numnber of them will be combined together
                q_ttp[0] += sent # The products need to be sent will be updated in the 0-level of q_ttp
                
       # Send Update
            # print("\n#Send Update")
            for i in range(1,3):
                # print("i: ", i+1)
                sent = min(q_i[i-1], q_d[i]) # Check the demand from downstream and compare to the inventory to decide how many to sent out 
                # print("sent", sent)
                new_leadtime = None  
                if sent != 0:  
                    new_leadtime = __integerise__(leadtimes[i].generate())# Generate a radom leadtime for each echelon
                    # print("new_leadtime: ", new_leadtime)
                    q_d[i] -= sent  # If the products have been sent, the vector of demand can be updated
                    q_i[i-1] -= sent   # The inventory of this echelon then will be updated
                    if new_leadtime not in q_p[i]: # Same as the steps we have done in supply update, but this will be processed in other echelon
                        q_p[i][new_leadtime] = sent
                    else:
                        q_p[i][new_leadtime] += sent
                    q_ttp[i] += sent
                # print("self.q_p: ", self.q_p)
                # print("self.q_ttp: ", self.q_ttp)
         
            # Client Demand Update\
            # print("\n#Client Demand Update")
            time_step += 1
            new_demand = __integerise__(demand.generate())
            q_d[-1] += new_demand
            # print("\nnew_demand",new_demand,"\nself.q_d: ", self.q_d)
                
                     # Trade Update
            # print("\n#Trade Update")
            amount = min(q_i[-1],q_d[-1])
            gain = price * amount
            reward += gain
            info['P'] += gain
            q_i[-1] -= amount
            q_d[-1] -= amount # Demand and inventory of retailer have been updated
            # print("amount", amount, "\nself.q_i: ", self.q_i, "\nself.q_d: ", self.q_d)       
                
                
action = np.zeros(shape = (3,), dtype = int) # Reset the action




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
epsilon = 0.9
        for i in range((10)+1):
            for j in range ((10)+1):
                for k in range ((10)+1):
                    actions_list[number_actions]=np.array([i,j,k])
                    actions.append(number_actions)
                    number_actions+=1
        q_table = pd.DataFrame(columns=actions, dtype=np.float64)
        eligibility_trace = q_table.copy()
        
next_state={"q_i": [0, 0, 0], "q_ttp": [0, 0, 0], "q_d": [0, 0, 0, 0], "q_p": [{}, {}, {}]}
next_state=json.dumps(next_state,cls=NumpyEncoder)
state=json.dumps(state,cls=NumpyEncoder)
        if next_state not in q_table.index:
            print("stpe_1")
                # append new state to q table
            to_be_append=pd.Series(
                    [0]*len(actions),
                    index=q_table.columns,
                    name=state,)
            q_table = q_table.append(to_be_append)
            states_list[number_states]=state
            # append state value to the states_list
            # also update eligibility trace
            eligibility_trace =eligibility_trace.append(to_be_append)
           
        key_list = list(states_list.keys()) 
        val_list = list(states_list.values())
        # Reture the key in the dictionary
        return key_list[val_list.index(state)]

            
    
        # action selection
        if np.random.rand() < epsilon:
            # choose best action
            state_action = q_table.loc[state,:]
            # some actions may have the same value, randomly choose on in these actions
            action_key = np.random.choice(state_action[state_action == np.max(state_action)].index)
            action=actions_list[action_key]
        else:
            # choose random action
            action = env.random_action()
        return action





state=env.reset()

agent.train()

next_state, reward, done, info = env.step(action)    
 
agent.state = env.reset()  
agent.next=next_state
        agent.check_state_exist(agent.state)
        agent.state=json.dumps(agent.state,cls=NumpyEncoder)
        q_predict = agent.q_table.loc[agent.state, action_key] # q_predict is the reward you get at the moment
        # Method 2:
        agent.eligibility_trace.loc[agent.state, :] *= 0
        agent.eligibility_trace.loc[agent.state, agent.action_key] = 1
        agent
        type(agent.next)
        agent.check_state_exist(agent.next)
        agent.next_action=agent.predict(agent.next)
        self.next=json.dumps(self.next,cls=NumpyEncoder)
        if self.done != True:
            q_target = self.reward + self.gamma * self.q_table.loc[self.next, self.action_key]  # next state is not terminal
        else:
            q_target = self.reward  # next state is terminal
        error = q_target - q_predict
        self.td_error+=error
        # increase trace amount for visited state-action pair

        # Method 1:
        # self.eligibility_trace.loc[s, a] += 1


        # Q update
        self.q_table += self.lr * error * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma*self.lambda_      

import matplotlib.pyplot as plt

print(plt.rcParams.get('figure.figsize')) # Check the default size

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size # Change the default size from[6,4] to [10,8]

x = np.linspace(-10, 9, 20)

y = x ** 3

plt.plot(x, y, 'b')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Cube Function')
plt.show()

x=1
y=x ** 3
plt.plot(x, y, 'b')
x=2
y=x ** 3
plt.plot(x, y, 'b')


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


expected = np.arange(100, dtype=np.float)
dumped = json.dumps(expected, cls=NumpyEncoder)
result = json.loads(dumped, object_hook=json_numpy_obj_hook)


# None of the following assertions will be broken.
assert result.dtype == expected.dtype, "Wrong Type"
assert result.shape == expected.shape, "Wrong Shape"
assert np.allclose(expected, result), "Wrong Values"





from Agents.SarsaLamdaAgent import SarsaLamdaAgent
from SupplyChain.SequentialSupplyChain import SequentialSupplyChain
from Agents.BenchmarkAgent import BenchmarkAgent

NUM_EPISODES = 50
LEN_EPISODE  = lambda x: 100
DECISION_INTERVAL = 5
AGENT_CLASS = SarsaLamdaAgent

agent_class = AGENT_CLASS
env = SequentialSupplyChain()
ag = BenchmarkAgent(env)
agent = agent_class(env)
reward_history = list()
cum_reward = 0
cum_info = {'c_i':0, 'c_p':0, 'c_d':0, 'c_o':0, 'P':0}
state = env.reset()
abs_sug = np.zeros(shape=(env.n,))
actions_list={}
actions=[]
actions_list={}
number_actions=1
action_key=0
gamma=0.9
lr=0.01
lambda_ =0.9

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
params = {
    'learning_rate':0.01,
    'reward_decay':0.9,
    'e_greedy':0.9,
    'trace_decay':0.9,
    'maximal_action':10
}
agent.__init__(env,params)

agent.check_state_exist(agent.state)
agent.state=json.dumps(agent.state,cls=NumpyEncoder)
agent.q_table.index
action=agent.predict(agent.state)

next_state, reward, done, info = env.step(action)

agent.witness(state, action, next_state, reward, done ) 
agent.train()
type(agent.next)
agent.check_state_exist(agent.next)
agent.next=str(agent.next)
q_predict = agent.q_table.loc[agent.state, agent.action_key] # q_predict is the reward you get at the moment
        # Method 2:
agent.eligibility_trace.loc[agent.state, :] *= 0
agent.eligibility_trace.loc[agent.state, agent.action_key] = 1
agent.check_state_exist(agent.next_state)
        agent.next_state=json.dumps(agent.next_state,cls=NumpyEncoder)

        if self.done != True:
            q_target = self.reward + self.gamma * self.q_table.loc[self.next, self.action_key]  # next state is not terminal
        else:
            q_target = self.reward  # next state is terminal
        error = q_target - q_predict
        self.td_error+=error
        # increase trace amount for visited state-action pair

        # Method 1:
        # self.eligibility_trace.loc[s, a] += 1


        # Q update
        self.q_table += self.lr * error * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma*self.lambda_  






         