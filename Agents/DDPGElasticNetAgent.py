"""
    This script implements a deep deterministic policy gradient agent with elastic net as  for sequential supply chain model, using TensorFlow.
"""

import numpy as np
import tensorflow as tf
import math
import random
from collections import deque

from .BenchmarkAgent import BenchmarkAgent
from . import utils

### Actor Structure [num_neurones, num_layers(residual)]
UNITS_A = [ 8, 3 ]

## Critic Structure
UNITS_C = [ 8, 3 ]

# Regularization
L2 = .001
L1 = .001

# Memory
MEMORY_SIZE = 20000

# Batch
BATCH_SIZE = 64
NUM_BATCH = 5

# Learning Rate
LR_A = 1e-5
LR_C = 1e-4

# Decau Schema
DECAY_RATE = 0.
DECAY_STEPS = 10

# Discount Rate
DISCOUNT_RATE = 0.

# Target Update Rate
TARGET_UPDATE_RATE = .1

# Dropout Rate
DROPOUT_RATE = 0.

# Noise
NOISE_SCALE = 1.
NOISE_DECAY_RATE = .99
NOISE_DECAY_STEPS = 1

# Max State
MAXS = 100

class DDPGElasticNetAgent:

    def __init__( self, env, params ):

        ## Environment Meta Parameters
        self.env = env
        self.N = self.env.N
        self.maxA = self.env.maxA
        self.maxS = MAXS

        ## Decision Interval Length
        self.decision_interval = params['decision_interval']

        ## Agent that Proposes Solutions
        self.prophet = BenchmarkAgent( env, params )

        self.inspected = dict()
        self.inspected_values = dict()

        self._build_network()
        self.sess = tf.compat.v1.Session()
        self.summary = tf.compat.v1.summary.FileWriter( './summary', self.sess.graph )

        self.sess.run( tf.compat.v1.global_variables_initializer() )

        self.memory = deque( maxlen = MEMORY_SIZE )

    def restore( self, checkpoint_file):

        saver = tf.compat.v1.train.Saver()
        saver.restore( self.sess, checkpoint_file )

    def _build_network( self ):

        ## Placeholders
        self.global_step = tf.Variable( initial_value = 0, trainable = False,
                                        name = 'global_step', dtype = tf.int32 )
        self.state_ph = tf.compat.v1.placeholder( dtype = tf.float32, shape = [None, self.N * 4 + 1], name = 'state_ph' )
        self.action_ph = tf.compat.v1.placeholder( dtype = tf.float32, shape = [None, self.N], name = 'action_ph' )
        self.next_state_ph = tf.compat.v1.placeholder( dtype = tf.float32, shape = [None, self.N * 4 + 1], name = 'next_state_ph' )
        self.reward_ph = tf.compat.v1.placeholder( dtype = tf.float32, shape = [None, 1], name = 'reward_ph' )
        self.is_training_ph = tf.compat.v1.placeholder( dtype = tf.float32, shape = [], name = 'is_training_ph' )

        ## Learning Schema
        self.dropout_rate = tf.compat.v1.train.inverse_time_decay(
            learning_rate = DROPOUT_RATE,
            global_step = self.global_step,
            decay_steps = DECAY_STEPS,
            decay_rate = DECAY_RATE,
            name = 'dropout_rate' )
        self.actor_decayed_learning_rate = tf.compat.v1.train.inverse_time_decay(
            learning_rate = LR_A,
            global_step = self.global_step,
            decay_steps = DECAY_STEPS,
            decay_rate = DECAY_RATE,
            name = 'actor_decayed_learning_rate' )
        self.critic_decayed_learning_rate = tf.compat.v1.train.inverse_time_decay(
            learning_rate = LR_C,
            global_step = self.global_step,
            decay_steps = DECAY_STEPS,
            decay_rate = DECAY_RATE,
            name = 'critic_decayed_learning_rate' )
        self.target_update_rate = TARGET_UPDATE_RATE
        self.discount_rate = DISCOUNT_RATE
        self.noise = tf.Variable( np.zeros(shape = (1,self.N) ), dtype = tf.float32 )
        self.noise_scale = tf.compat.v1.train.inverse_time_decay(
            learning_rate = NOISE_SCALE,
            global_step = self.global_step,
            decay_steps = NOISE_DECAY_STEPS,
            decay_rate = NOISE_DECAY_RATE,
            name = 'decayed_noise_scale' )

        ## Noise Generation and Annihilation (when using Brownian noise, set noise process to zero)
        self.noise_generation_op = tf.compat.v1.assign( ref = self.noise,
            value = tf.random.normal( shape = [1, self.N], dtype = tf.float32) * self.noise_scale )
        self.noise_annihilation_op = tf.compat.v1.assign( ref = self.noise, value = np.zeros( shape = (1,self.N) ))

        ## Actor and Critic Behavior and Target
        with tf.compat.v1.variable_scope('behavior_actor'):
            self.predicted_action = self._build_actor( self.state_ph )
        with tf.compat.v1.variable_scope('target_actor'):
            self.predicted_next_action = self._build_actor( self.next_state_ph )
        with tf.compat.v1.variable_scope('behavior_critic'):
            self.predicted_given_action_value = self._build_critic( self.state_ph, self.action_ph )
        with tf.compat.v1.variable_scope('behavior_critic', reuse = True):
            self.predicted_best_action_value = self._build_critic( self.state_ph, self.predicted_action )
        with tf.compat.v1.variable_scope('target_critic'):
            self.predicted_next_action_value = self._build_critic( self.next_state_ph, self.predicted_next_action )

        ## Retrieve lists of weights
        self.behavior_actor_weights = tf.compat.v1.get_collection( tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='behavior_actor' )
        self.target_actor_weights = tf.compat.v1.get_collection( tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor' )
        self.behavior_critic_weights = tf.compat.v1.get_collection( tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='behavior_critic' )
        self.target_critic_weights = tf.compat.v1.get_collection( tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic' )

        ## Build Train Operations
        self._build_train_ops()

    def _build_train_ops( self ):

        # increment global step
        self.increment_global_step = tf.compat.v1.assign_add( self.global_step, 1, name = 'increment_global_step' )

        # loss of behavior actor ( -reward )
        self.behavior_actor_loss = - tf.reduce_mean( input_tensor=self.predicted_best_action_value )
        for weight in self.behavior_actor_weights:
            if not 'bias' in weight.name:
                self.behavior_actor_loss += .5 * L2 * tf.nn.l2_loss( weight )
        self.behavior_actor_train_op = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate = self.actor_decayed_learning_rate).minimize(
                loss = self.behavior_actor_loss,
                var_list = self.behavior_actor_weights,
                name = 'behavior_actor_train_op' )

        # loss of behavior critic ( td error )
        td_errors = tf.add( self.reward_ph \
            + self.discount_rate * self.predicted_next_action_value, \
            - self.predicted_given_action_value, name = 'td_errors')
        self.inspect(tf.reduce_mean(input_tensor=tf.abs(td_errors)),'td_errors')
        self.inspect(tf.reduce_mean(input_tensor=self.predicted_given_action_value),'predicted_action_values')

        self.behavior_critic_loss = tf.reduce_mean( input_tensor=tf.square(td_errors) )
        for weight in self.behavior_critic_weights:
            if not 'bias' in weight.name:
                print(weight.shape)
                self.behavior_critic_loss += .5 * L2 * tf.nn.l2_loss( weight ) \
                        + L1 * tf.reduce_mean( input_tensor=tf.abs( weight ) )
        self.behavior_critic_train_op = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate = self.critic_decayed_learning_rate).minimize(
                loss = self.behavior_critic_loss,
                var_list = self.behavior_critic_weights,
                name = 'behavior_critic_train_op' )
        self.inspect( tf.reduce_mean(input_tensor=self.behavior_critic_loss),'critic_loss' )

        # update target networks
        target_update_ops = list()
        for w_b, w_t in zip(self.behavior_actor_weights,self.target_actor_weights):
            target_update_ops.append( w_t.assign( self.target_update_rate * w_b
                                                + (1.-self.target_update_rate) * w_t ) )
        for w_b, w_t in zip(self.behavior_critic_weights,self.target_critic_weights):
            target_update_ops.append( w_t.assign( self.target_update_rate * w_b
                                                + (1.-self.target_update_rate) * w_t ) )
        self.target_update_ops = tf.group( *target_update_ops, name = 'target_update_ops' )

        # copy target to behavior
        behavior_copy_ops = list()
        for w_b, w_t in zip(self.behavior_actor_weights,self.target_actor_weights):
            target_update_ops.append( w_b.assign( w_t ) )
        for w_b, w_t in zip(self.behavior_critic_weights,self.target_critic_weights):
            target_update_ops.append( w_b.assign( w_t ) )
        self.behavior_copy_ops = tf.group( *behavior_copy_ops, name = 'behavior_copy_ops' )

    def inspect( self, tensor, name ):
        self.inspected[name] = tensor
        self.inspected_values[name] = list()

    def _build_hidden_layer( self, inputs, units ):

        x = tf.compat.v1.layers.dense(
            inputs = inputs,
            units = units,
            activation = tf.nn.relu,
            kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            name = 'dense' )
        # return x
        # x = tf.layers.dropout( hid, rate = self.dropout_rate,
                                # name = 'dropout' )
        x = tf.compat.v1.layers.dense(
            inputs = x,
            units = units,
            kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            name = 'dense_linear' )
        out = tf.nn.relu(inputs + x)
        return out

    def _build_actor_output_layer( self, inputs ):

        x = inputs
        x = tf.compat.v1.layers.dense(
            inputs = x,
            kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            units = self.N)
        x = tf.add( x, self.is_training_ph * self.noise, name = 'output' )
        return x

    def _build_first_hidden_layer( self, inputs, units ):

        return tf.compat.v1.layers.dense(
            inputs = inputs,
            kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            units = units )

    def _build_actor( self, state ):

        x = state
        x = self._build_first_hidden_layer( x, UNITS_A[0] )
        for i in range(UNITS_A[1]):
            with tf.compat.v1.variable_scope('hidden'+str(i)):
                x = self._build_hidden_layer( x, UNITS_A[0] )
        with tf.compat.v1.variable_scope('output'):
            x = self._build_actor_output_layer( x )
        x = tf.clip_by_value( x + state[:,-self.N:], clip_value_min = 0., clip_value_max = self.maxA )
        return x

    def _build_critic( self, state, action ):

        x = tf.concat( values = [state, action], axis = 1, name = 'concatenated_input' )
        x = tf.compat.v1.layers.dense(
            inputs = x,
            units = 1,
            kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
            name = 'output')
        return x

    def _preprocess_state( self, state ):

        prophet_action = self._preprocess_action( self.prophet.predict( state ) )
        processed_state = np.concatenate( [state['q_i'],state['q_ttp'],state['q_d']], axis=0 ).astype(float).clip(min=0,max=self.maxS)/self.maxS
        return np.concatenate( [processed_state,prophet_action], axis=0 )

    def _preprocess_action( self, action ):
        return action.astype(float).clip(min=0,max=self.maxA)/self.maxA

    def _postprocess_action( self, action ):
        return utils.integerise( action * self.maxA )

    def predict( self, state, is_training = True ):

        state = np.expand_dims( self._preprocess_state( state ), axis=0 )
        action = self.sess.run( [self.predicted_action],
            feed_dict = { self.state_ph: state, self.is_training_ph: float(is_training) } )[0][0]

        return self._postprocess_action( action )

    def witness( self, state, action, next_state, reward, done ):

        if done:
            self.sess.run([self.noise_annihilation_op])

        state = self._preprocess_state( state )
        action = self._preprocess_action( action )
        next_state = self._preprocess_state( next_state )
        reward = np.expand_dims( reward, axis=0 ).astype(float)

        self.memory.append( [state, action, next_state, reward] )

    def train( self ):

        inspected_values = None

        for ep in range(NUM_BATCH):

            batch_size = min( len(self.memory), BATCH_SIZE )
            batch = random.sample( self.memory, batch_size )

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

            self.sess.run([self.noise_generation_op])

            inspected_value, *rest = self.sess.run([
                    self.inspected,
                    self.behavior_actor_train_op,
                    self.behavior_critic_train_op],
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
            self.inspected_values[k].append(np.mean(inspected_value[k]))
