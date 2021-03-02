"""
    This script implements a Deep Q-Network agent for sequential supply chain model, using TensorFlow.
"""

import numpy as np
import tensorflow as tf
import math
import random
from collections import deque
from tensorflow.python import debug as tf_debug

from .DemandAgent import DemandAgent
from .BenchmarkAgent import BenchmarkAgent
from . import standardisers as S
from . import features as F
from . import utils as U

tf.compat.v1.disable_eager_execution()

params = {
    'actor':{
        'num_neurones':64,
        'num_layers':5
    },
    'critic':{
        'num_neurones':64,
        'num_layers':5
    },
    'l2':.01,
    'l1':.01,
    'memory_size':2000,
    'batch_size':64,
    'num_batches':4,
    'learning_rate':{
        'actor':1e-4,
        'critic':1e-4,
        'entropy':1e-5,
    },
    'decay':{
        'rate':.1,
        'steps':50
    },
    'discount':1.-1e-2,
    'target_update_rate':.1,
    'dropout_rate':1.,
    'noise_log_std':1e-2,
    'feature_state':'beer_game_state',
    'standardisers':{
        'target_std':.01,
        'centralisation':True,
        'normalisation':True,
    },
    'action_range':{
        'min':-2,
        'max':11,
    },
    'prophet_class':BenchmarkAgent,
}

class DQNAgent:

    def __init__(self, env, params=params):

        ## Environment Meta Parameters
        self.n = env.n
        self.maximal_action = env.maximal_action
        self.decision_interval = env.decision_interval

        ## Agent that Proposes Solutions
        self.prophet = params['prophet_class'](env)

        self.inspected = dict()
        self.inspected_values = dict()

        with tf.compat.v1.variable_scope('dqn'):
            self.__build_network__(params)
            self.sess = tf.compat.v1.Session()
            self.summary = tf.compat.v1.summary.FileWriter('./summary', self.sess.graph)

            self.sess.run(tf.compat.v1.global_variables_initializer())

        self.memory = deque(maxlen = params['memory_size'])

    def restore(self, checkpoint_file):

        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, checkpoint_file)

    def __build_network__(self, params):

        self.feature_dim = F.feature_dims[params['feature_state']](self.n)
        self.feature_transformation = F.feature_transformations[params['feature_state']]
        self.action_range = params['action_range']
        self.action_range['len'] = self.action_range['max']-self.action_range['min']+1

        ## Placeholders
        self.global_step = tf.Variable(initial_value = 0, trainable = False,
                                        name = 'global_step', dtype = tf.int32)
        self.state_ph = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, self.n+self.feature_dim], name = 'state_ph')
        self.action_ph = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, self.n*self.action_range['len']], name = 'action_ph')
        self.next_state_ph = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, self.n+self.feature_dim], name = 'next_state_ph')
        self.reward_ph = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, 1], name = 'reward_ph')
        self.is_training_ph = tf.compat.v1.placeholder(dtype = tf.float32, shape = [], name = 'is_training_ph')

        ## Learning Schema
        learning_rate = params['learning_rate']
        decay = params['decay']
        self.dropout_rate = params['dropout_rate']

        self.actor_decayed_learning_rate = tf.compat.v1.train.inverse_time_decay(
            learning_rate = learning_rate['actor'],
            global_step = self.global_step,
            decay_steps = decay['steps'],
            decay_rate = decay['rate'],
            name = 'actor_decayed_learning_rate')
        self.critic_decayed_learning_rate = tf.compat.v1.train.inverse_time_decay(
            learning_rate = learning_rate['critic'],
            global_step = self.global_step,
            decay_steps = decay['steps'],
            decay_rate = decay['rate'],
            name = 'critic_decayed_learning_rate')
        self.target_update_rate = params['target_update_rate']
        self.discount_rate = params['discount']
        self.entropy_learning_rate = tf.compat.v1.train.inverse_time_decay(
            learning_rate = learning_rate['entropy'],
            global_step = self.global_step,
            decay_steps = decay['steps'],
            decay_rate = decay['rate'],
            name = 'decayed_entropy_learning_rate')

        self.noise_log_std = tf.compat.v1.train.inverse_time_decay(
            learning_rate = params['noise_log_std'],
            global_step = self.global_step,
            decay_steps = decay['steps'],
            decay_rate = decay['rate'],
            name = 'decayed_entropy_learning_rate')
        self.noise_std = tf.compat.v1.train.inverse_time_decay(
            learning_rate = params['noise_log_std'],
            global_step = self.global_step,
            decay_steps = decay['steps'],
            decay_rate = decay['rate'],
            name = 'decayed_entropy_learning_rate')
        self.noise = tf.random.normal(shape=(tf.shape(input=self.state_ph)[0],self.n), mean=0.0, stddev=self.noise_std, dtype=tf.float32)
        # self.noise_log_std = tf.zeros(shape = (1,self.n), dtype = tf.float32) + params['noise_log_std']

        standardisers_params = params['standardisers']
        with tf.compat.v1.variable_scope('state_standardiser'):
            self.state_standardiser = S.Standardiser(self.state_ph, params=standardisers_params)
            self.standardised_state = self.state_standardiser.output
        with tf.compat.v1.variable_scope('state_standardiser', reuse = True):
            self.next_state_standardiser = S.Standardiser(self.next_state_ph, params=standardisers_params)
            self.preprocessed_next_state = self.next_state_standardiser.output
        with tf.compat.v1.variable_scope('reward_standardiser'):
            self.reward_standardiser = S.Standardiser(self.reward_ph, params=standardisers_params)
            self.standardised_reward = self.reward_standardiser.output

        ## Actor and Critic Behavior and Target
        self.actor_meta = params['actor']
        self.critic_meta = params['critic']
        with tf.compat.v1.variable_scope('behavior_actor'):
            self.predictions = self.__build_actor__(self.state_ph)
        with tf.compat.v1.variable_scope('target_actor'):
            self.next_predictions = self.__build_actor__(self.next_state_ph)
        with tf.compat.v1.variable_scope('behavior_critic'):
            self.predicted_given_action_value = self.__build_critic__(self.state_ph, self.action_ph)
        with tf.compat.v1.variable_scope('behavior_critic', reuse = True):
            self.predicted_best_action_value = self.__build_critic__(self.state_ph, self.predictions['probs'])
        with tf.compat.v1.variable_scope('target_critic'):
            self.predicted_next_action_value = self.__build_critic__(self.next_state_ph, self.next_predictions['probs'])

        ## Retrieve lists of weights
        self.behavior_actor_weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='behavior_actor')
        self.target_actor_weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')
        self.behavior_critic_weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='behavior_critic')
        self.target_critic_weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')

        ## Build Train Operations
        self.l2 = params['l2']
        self.l1 = params['l1']
        self.num_batches = params['num_batches']
        self.batch_size = params['batch_size']
        self.__build_train_ops__()

    def __build_train_ops__(self):

        # increment global step
        self.increment_global_step = tf.compat.v1.assign_add(self.global_step, 1, name = 'increment_global_step')

        # loss of behavior actor (-reward)
        self.behavior_actor_loss = - tf.reduce_mean(input_tensor=self.predicted_best_action_value)
        for weight in self.behavior_actor_weights:
            if not 'bias' in weight.name:
                self.behavior_actor_loss += .5 * self.l2 * tf.nn.l2_loss(weight)
                self.behavior_actor_loss += self.l1 * tf.reduce_mean(input_tensor=tf.abs(weight))
        self.entropy = tf.reduce_sum(input_tensor=self.noise_log_std)
        # self.behavior_actor_loss -= self.entropy_learning_rate * self.entropy
        self.behavior_actor_train_op = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate = self.actor_decayed_learning_rate).minimize(
                loss = self.behavior_actor_loss,
                var_list = self.behavior_actor_weights,
                name = 'behavior_actor_train_op')
        self.inspect(self.entropy, 'entropy')
        self.inspect(self.behavior_actor_loss, 'actor_loss')

        # loss of behavior critic (td error)
        td_errors = tf.add(self.standardised_reward \
            + self.discount_rate * self.predicted_next_action_value, \
            - self.predicted_given_action_value, name = 'td_errors')
        self.inspect(tf.reduce_mean(input_tensor=tf.abs(td_errors)),'td_errors')
        self.inspect(tf.reduce_mean(input_tensor=self.predicted_given_action_value),'predicted_action_values')

        # Entropy training
        self.entropy = - tf.reduce_mean(input_tensor=self.predictions['entropy'])
        self.entropy_train_op = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate = self.actor_decayed_learning_rate).minimize(
                loss = self.behavior_actor_loss,
                var_list = self.behavior_actor_weights,
                name = 'behavior_actor_train_op')
        self.inspect(self.entropy, 'entropy')

        self.behavior_critic_loss = tf.reduce_mean(input_tensor=tf.square(td_errors))
        for weight in self.behavior_critic_weights:
            if not 'bias' in weight.name:
                self.behavior_critic_loss += .5 * self.l2 * tf.nn.l2_loss(weight)
                self.behavior_critic_loss += self.l1 * tf.reduce_mean(input_tensor=tf.abs(weight))
        self.behavior_critic_train_op = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate = self.critic_decayed_learning_rate).minimize(
                loss = self.behavior_critic_loss,
                var_list = self.behavior_critic_weights,
                name = 'behavior_critic_train_op')
        self.inspect(tf.reduce_mean(input_tensor=self.behavior_critic_loss),'critic_loss')

        # update target networks
        target_update_ops = list()
        for w_b, w_t in zip(self.behavior_actor_weights,self.target_actor_weights):
            target_update_ops.append(w_t.assign(self.target_update_rate * w_b
                                                + (1.-self.target_update_rate) * w_t))
        for w_b, w_t in zip(self.behavior_critic_weights,self.target_critic_weights):
            target_update_ops.append(w_t.assign(self.target_update_rate * w_b
                                                + (1.-self.target_update_rate) * w_t))
        self.target_update_ops = tf.group(*target_update_ops, name = 'target_update_ops')

        # copy target to behavior
        behavior_copy_ops = list()
        for w_b, w_t in zip(self.behavior_actor_weights,self.target_actor_weights):
            target_update_ops.append(w_b.assign(w_t))
        for w_b, w_t in zip(self.behavior_critic_weights,self.target_critic_weights):
            target_update_ops.append(w_b.assign(w_t))
        self.behavior_copy_ops = tf.group(*behavior_copy_ops, name = 'behavior_copy_ops')

    def inspect(self, tensor, name):
        self.inspected[name] = tensor
        self.inspected_values[name] = list()

    def __build_hidden_layer__(self, inputs, units):

        x = tf.compat.v1.layers.dense(
            inputs = inputs,
            units = units,
            activation = tf.nn.relu,
            kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            name = 'dense')
        x = tf.compat.v1.layers.dropout(x, rate = self.dropout_rate, name = 'dropout')
        return x

    def __build_actor_output_layer__(self, inputs):

        x = inputs

        min_action = self.action_range['min']
        max_action = self.action_range['max']
        num_actions = max_action - min_action + 1

        logits = []
        for i in range(self.n):
            logits.append(tf.compat.v1.layers.dense(
                inputs = x,
                kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                units = num_actions))
        probs = [ tf.nn.softmax(logit) for logit in logits ]
        samples = [ tf.random.categorical(logits=logit,num_samples=1) for logit in logits ]
        one_hots = [ tf.one_hot(sample, num_actions) for sample in samples ]
        actions = [ sample + min_action for sample in samples ]

        logits = tf.concat(logits,axis=-1)
        probs = tf.concat(probs,axis=-1)
        one_hots = tf.cast(tf.concat(one_hots,axis=-1), tf.float32)
        actions = tf.cast(tf.concat(actions,axis=-1), tf.float32)

        return {'probs':probs,'one_hots':one_hots,'actions':actions,'entropy':-tf.reduce_sum(input_tensor=logits*probs,axis=-1)}

    def __build_first_hidden_layer__(self, inputs, units):

        return tf.compat.v1.layers.dense(
            inputs = inputs,
            kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            units = units)

    def __build_actor__(self, state):

        x = state
        x = self.__build_first_hidden_layer__(x, self.actor_meta['num_neurones'])
        for i in range(self.actor_meta['num_layers']):
            with tf.compat.v1.variable_scope('hidden'+str(i)):
                x = self.__build_hidden_layer__(x, self.actor_meta['num_neurones'])
        with tf.compat.v1.variable_scope('output'):
            x = self.__build_actor_output_layer__(x)
        x['actions'] = tf.clip_by_value(x['actions'] + state[:,-self.n:], clip_value_min = 0., clip_value_max = float('inf'))
        return x

    def __build_critic_output_layer__(self, inputs):
        x = tf.nn.tanh(inputs)
        x = tf.compat.v1.layers.dense(
            inputs = x,
            units = 1,
            kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            name = 'output')
        return x

    def __build_critic__(self, state, action):

        x = tf.concat(values = [state, action], axis = 1, name = 'concatenated_input')
        x = self.__build_first_hidden_layer__(x, self.critic_meta['num_neurones'])
        for i in range(self.critic_meta['num_layers']):
            with tf.compat.v1.variable_scope('hidden'+str(i)):
                x = self.__build_hidden_layer__(x, self.critic_meta['num_neurones'])
        with tf.compat.v1.variable_scope('output'):
            x = self.__build_critic_output_layer__(x)
        return x

    def __preprocess_state__(self, state):

        prophet_action = self.prophet.predict(state)
        processed_state = self.feature_transformation(state, self.decision_interval)
        return np.concatenate([processed_state,prophet_action], axis=0)

    def __preprocess_action__(self, action):
        return np.concatenate(U.np_one_hot(action - self.action_range['min'], self.action_range['len']), axis=-1)

    def __postprocess_action__(self, action):
        return action * self.maximal_action

    def predict(self, state, is_training = True):

        state = np.expand_dims(self.__preprocess_state__(state), axis=0)
        predictions = self.sess.run(self.predictions,
            feed_dict = { self.state_ph: state, self.is_training_ph: float(is_training) })
        return predictions['actions'][0]

    def witness(self, state, action, next_state, reward, done):

        state = self.__preprocess_state__(state)
        action -= state[-self.n:]
        action = self.__preprocess_action__(action)
        next_state = self.__preprocess_state__(next_state)
        reward = np.expand_dims(reward, axis=0).astype(float)

        self.memory.append([state, action, next_state, reward])

    def train(self):

        inspected_values = None

        for ep in range(self.num_batches):

            batch_size = min(len(self.memory), self.batch_size)
            batch = random.sample(self.memory, batch_size)

            state, action, next_state, reward = [], [], [], []
            for sample in batch:
                cs,a,ns,r = sample
                state.append(cs)
                action.append(a)
                next_state.append(ns)
                reward.append(r)
            state = np.array(state)
            action = np.array(action)
            next_state = np.array(next_state)
            reward = np.array(reward)

            inspected_values, *rest = self.sess.run([
                    self.inspected,
                    self.behavior_actor_train_op,
                    self.behavior_critic_train_op,
                    self.entropy_train_op],
                feed_dict = {
                    self.state_ph: state,
                    self.action_ph: action,
                    self.next_state_ph: next_state,
                    self.reward_ph: reward,
                    self.is_training_ph: 1.
                })

        self.sess.run([self.target_update_ops])
        self.sess.run([self.behavior_copy_ops])
        self.sess.run([self.increment_global_step])

        for k in self.inspected:
            self.inspected_values[k].append(np.mean(inspected_values[k]))
