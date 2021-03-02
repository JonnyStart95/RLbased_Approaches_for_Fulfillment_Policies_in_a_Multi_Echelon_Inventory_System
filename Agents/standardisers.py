import numpy as np
import warnings

EPS = 1e-3
import tensorflow as tf

params = {
    'centralisation':True,
    'normalisation':True,
    'target_std':1.,
}

class Standardiser:

    def __init__( self, tensor, params=params ):

        self.eps = EPS
        self.centralisation = params['centralisation']
        self.normalisation = params['normalisation']
        self.target_std = params['target_std']

        self.tensor = tensor
        self.shape = tensor.shape
        self.means = tf.compat.v1.get_variable( 'means', shape=self.shape[1:], dtype=tf.float32, initializer=tf.compat.v1.zeros_initializer() )
        self.means_sq = tf.compat.v1.get_variable( 'means_sq', shape=self.shape[1:], dtype=tf.float32, initializer=tf.compat.v1.ones_initializer() )
        self.vars = tf.compat.v1.get_variable( 'vars', shape=self.shape[1:], dtype=tf.float32, initializer=tf.compat.v1.ones_initializer() )
        self.count = tf.compat.v1.get_variable( 'count', shape=(), dtype=tf.float32, initializer=tf.compat.v1.zeros_initializer() )

        # self.means = tf.Variable( np.zeros(shape=self.shape[1:]), dtype = tf.float32, name = 'means' )
        # self.means_sq = tf.Variable( np.ones(shape=self.shape[1:]), dtype = tf.float32, name = 'means_sq' )
        # self.vars = tf.Variable( np.ones(shape=self.shape[1:]), dtype = tf.float32, name = 'vars' )
        # self.count = tf.Variable( 0, dtype = tf.float32, name = 'count' )

        self.new_count = tf.cast( tf.shape(input=self.tensor)[0], tf.float32 )

        total_count = self.new_count + self.count
        old_percentage = self.count / total_count
        new_means = self.means * old_percentage + tf.reduce_mean(input_tensor=self.tensor,axis=0) * (1.-old_percentage)
        new_means_sq = self.means_sq * old_percentage + tf.reduce_mean(input_tensor=self.tensor**2.,axis=0) * (1.-old_percentage)

        self.vars_update_op = tf.compat.v1.assign( self.vars, self.vars + new_means_sq - self.means_sq - (new_means**2.-self.means**2.) )
        with tf.control_dependencies([self.vars_update_op]):
            self.means_update_op = tf.compat.v1.assign( self.means, new_means)
            self.means_sq_update_op = tf.compat.v1.assign( self.means_sq, new_means_sq)
            with tf.control_dependencies([self.means_update_op,self.means_sq_update_op]):
                self.count_update_op = tf.compat.v1.assign( self.count, total_count )
                with tf.control_dependencies([self.count_update_op]):
                    self.output = self.tensor
                    if self.centralisation:
                        self.output = self.output - self.means
                    if self.normalisation:
                        self.output = self.output / (self.vars ** .5 + self.eps) * self.target_std

        self.update_op = tf.group( [self.vars_update_op, self.means_update_op, self.means_sq_update_op, self.count_update_op] )

    def recover( self, standardised ):

        x = standardised / self.target_std
        if self.normalisation:
            x = x * self.vars ** .5
        if self.centralisation:
            x = x + self.means
        return x

    def as_func( self ):
        return {
            'tensor':self.tensor,
            'output':self.output,
            'train_op':self.train_op}

    def get_stats_tensor( self ):
        return{
            'means':self.means,
            'means_sq':self.means_sq,
            'vars':self.vars
        }

def test():

    length = 3
    means = np.array([[0.0,1.0,10.5]])
    stds = np.array([[100.,3.,0.05]])

    input_batches = tf.compat.v1.placeholder( shape=[None,length], dtype = tf.float32 )
    standardiser = Standardiser(input_batches)
    post_input, train_op = standardiser.output, standardiser.update_op

    e_means = standardiser.means
    e_stds = standardiser.vars ** .5

    sess = tf.compat.v1.Session()
    sess.run( tf.compat.v1.global_variables_initializer() )

    num_batches = 2000
    batch_size = 64
    for i in range(num_batches):
        inputs = np.random.randn(batch_size,length) * stds + means

        feed_dict = {input_batches:inputs}
        processed_results = sess.run(post_input,feed_dict=feed_dict)

        if (i+1) % 100 == 0:
            mus, sigmas, means_sq, count = sess.run([e_means,e_stds,standardiser.means_sq,standardiser.count])
            print('iterations: {}, mus error: {}, sigmas error: {}, means_sq: {}, count: {}'.format(
                    i+1, mus, sigmas * count / (count-1.), means_sq,count))

if __name__ == '__main__':
    test()
