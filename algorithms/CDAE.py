"""
@author : hyunsung lee
@email : ita9naiwa@gmail.com
"""
# coding: utf-8

from .Recommender import ImplicitRecommender
import time
import tensorflow as tf
import numpy as np 
from tqdm import tqdm
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

# steal from (tensorflow/somewhere/some_ops dropout)

def masking(x, keep_prob, noise_shape=None,noise_size = None, seed=None, name=None):
    # mask input with zero with probability 1 - keep_prob
    # Brought from "dropout(...)" in tensorflow repository
    with ops.name_scope(name, "masking", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        # Do nothing if we know keep_prob == 1
        
        if tensor_util.constant_value(keep_prob) == 1:
            return x
    
        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
    
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor
        return ret

class CDAE(ImplicitRecommender):
    
    def __init__(self,dtype = 'float32',verbose = True,seed=None,**kwargs):
        super(CDAE,self).__init__(dtype,verbose,seed,**kwargs)
        self.model_name = 'CDAE'
        self.sess = None
        self.seed = seed
        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self,**kwargs):
        self.latent_dim = int(kwargs.get('latent_dim',100))
        self.learning_rate = float(kwargs.get('leraning_rate',0.05))
        self.keep_prob = float(kwargs.get('keep_prob',0.5))
        self.use_user_bias = bool(kwargs.get('use_user_bias',False))
        self.lamb_total = float(kwargs.get('lambda_total',0.05))
        self.activation = str(kwargs.get('activation','sigmoid'))


    def _init_model(self,n_users,n_items):
        tf.reset_default_graph()
        tf.set_random_seed(self.seed)
        self.sess = tf.Session()
        latent_dim = self.latent_dim
        learning_rate = self.learning_rate

        with tf.name_scope("input"):
            self.x = tf.placeholder("float", [None,n_items],name='x')

            self.dropout_rate = tf.placeholder("float",name='dropout_rate')
            if self.use_user_bias:
                self.u = tf.placeholder("int32",[None])
                u = self.u

        # Corrupt input first
        self.x_tilda = masking(self.x,self.dropout_rate)

        # Can denote default initializer using variable scope like this

        regularizer = tf.contrib.layers.l2_regularizer(scale=self.lamb_total)
        with tf.variable_scope("parameter",initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=regularizer):
            self.w_in = tf.get_variable(name="w_in",shape = (n_items,latent_dim))
            self.w_out = tf.get_variable(name="w_out",shape = (latent_dim,n_items))
            self.b_in = tf.get_variable(name = "b_in",shape = (latent_dim,))
            self.b_out = tf.get_variable(name = "b_out",shape = (n_items,))

        
        x_tilda = self.x_tilda
        x = self.x
        w_in = self.w_in
        w_out = self.w_out
        b_in = self.b_in
        b_out = self.b_out

        trainable_parameters = [w_in,w_out,b_in,b_out]

        activation = None
        if self.activation == 'sigmoid':
            activation = tf.nn.sigmoid;
        elif self.activation =='tanh':
            activaton = tf.tanh
        elif self.activation == 'relu':
            activation = tf.nn.relu
        elif self.activation == 'identity':
            activation = lambda x : x


        if True == self.use_user_bias :
            self.user_bias = tf.get_variable(name = "user_bias",shape = (n_users,latent_dim))
            trainable_parameters.append(self.user_bias)
            hidden_layer = activation(tf.nn.bias_add(tf.gather(self.user_bias,u) + tf.matmul(x_tilda,w_in),b_in))
        else:
            hidden_layer = activation(tf.nn.bias_add(tf.matmul(x_tilda,w_in),b_in))
        hidden_layer = tf.nn.dropout(hidden_layer,self.dropout_rate)

        self.reconstructed_x = activation(tf.nn.bias_add(tf.matmul(hidden_layer,w_out),b_out))
    
        reconstructed_x = self.reconstructed_x

        self.pred_error = tf.reduce_mean(tf.reduce_sum((reconstructed_x-x)**2,axis=1),name = "mse")

        
        self.pred_error_with_reg = self.pred_error + tf.contrib.layers.apply_regularization(regularizer, trainable_parameters)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.pred_error_with_reg,
            var_list = trainable_parameters,name='opt')

        self.init = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
        self.sess.run(self.init)
        self.saver = tf.train.Saver()

        self.predicted = -(activation(tf.nn.bias_add(tf.matmul(activation(tf.nn.bias_add(tf.matmul(x,w_in),b_in)),w_out),b_out)))

    def train_model(self,X,n_iter = 10,batch_size = 16,dropout_rate = 0.5,vad_data = None,**kwargs):
        '''Fit the model to the interaction matrix X
        Parameters
        ----------
        X : ??
        vad_data = same data type and same shape with X;
        '''
        self.n_users,self.n_items = X.shape
        n_users,n_items = X.shape
        self._init_model(n_users,n_items)
        elapsed_time = self._update(X,vad_data,n_iter,batch_size,dropout_rate,**kwargs)
    
    def _update(self,X,vad_data,n_iter,batch_size,dropout_rate,**kwargs):

        if True == self.verbose:
            iter_state = tqdm(range(n_iter))
        else:
            iter_state = range(n_iter)

        begin_time = time.time()
        for _ in iter_state:
            cost_on_iteration = self.run_epoch(X,batch_size,dropout_rate)
            if self.verbose:
                print(self.info_per_iter(_,kwargs.get('test_visible',None),kwargs.get('test_hidden',None),kwargs.get('k',10)))
        return time.time() - begin_time
    

    def loss(self,X):
        if True == self.use_user_bias:
            feed_dict = {self.x : X,self.u : np.arange(X.shape[0])}
        else:
            feed_dict = {self.x : X}
        [loss] = self.sess.run([self.pred_error],feed_dict = feed_dict)
        return loss

    def predict(self,X):
        return np.asarray(self.sess.run(self.predicted,feed_dict = {self.x : X}))
    def predict_matrix(self):
        return self.predict(X)

    def run_epoch(self,X,batch_size,dropout_rate):
        costs = []
        n_rows = X.shape[0]
        count = n_rows // batch_size

        rows = np.random.choice(n_rows,size = (count,batch_size))
        for i in range(count):
            feed_dict = {self.x : X[rows[i]],self.dropout_rate : dropout_rate}
            if True == self.use_user_bias:
                feed_dict[self.u] = rows[i]

            _,current_cost_ = self.sess.run([self.optimizer,self.pred_error],
            feed_dict = feed_dict)
            
            costs.append(current_cost_)
        return np.mean(costs)
    def predict_topk(self,X,k = 5):
        predicted = predict(X)
        return np.argpartition(predicted,k,axis=1)[:,:k]
    def predict_topk_test(self,X,X_test,k=5):
        predicted = np.asarray(self.sess.run(self.predicted,feed_dict = {self.x : X,self.dropout_rate : 1.0}))
        return np.argpartition(predicted,k,axis=1)[:,:k]


