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
from scipy.sparse import *
# steal from (tensorflow/somewhere/some_ops dropout)


def masking(x, keep_prob, noise_shape=None,noise_size = None, seed=None, name=None):
    # mask input with zero with probability 1 - keep_prob
    # Brought from "dropout(...)" in tensorflow repository
    one = tf.constant(1.0)
    with ops.name_scope(name, "masking", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        keep_prob = ops.convert_to_tensor(keep_prob, dtype="float", name="keep_prob")
        sz_f = tf.cast(tf.size(x),tf.float32)
        rv = tf.random_normal(mean = sz_f*keep_prob ,stddev = sz_f*(one-keep_prob)*keep_prob,shape = [])
        return tf.random_shuffle(x)[tf.cast(rv,tf.int32):]


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
            self.items = tf.placeholder("int32", [None,],name='items')
            self.negative_items = tf.placeholder("int32", [None,],name='negative_items')

            self.dropout_rate = tf.placeholder("float",name='dropout_rate')
            if self.use_user_bias:
                self.u = tf.placeholder("int32",[None])
                u = self.u
        # Corrupt input first
        # what about masking?
        self.x_tilda = masking(self.items,tf.constant(self.keep_prob))

        regularizer = tf.contrib.layers.l2_regularizer(scale=self.lamb_total)
        with tf.variable_scope("parameter",initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=regularizer):
            self.w_in = tf.get_variable(name="w_in",shape = (n_items,latent_dim))
            self.w_out = tf.get_variable(name="w_out",shape = (latent_dim,n_items))
            self.b_in = tf.get_variable(name = "b_in",shape = (latent_dim,))
            self.b_out = tf.get_variable(name = "b_out",shape = (n_items,))

        x_tilda = self.x_tilda
        x = self.items
        items =self.items
        negative_items = self.negative_items
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


        #hidden_layer = activation(tf.nn.bias_add(tf.sparse_tensor_dense_matmul(x_tilda,w_in),b_in))
        self.ret = tf.reduce_sum(tf.gather(w_in,items) ** 2)
        hidden_layer = activation(tf.reduce_sum(tf.gather(w_in,x_tilda),axis = 0) + b_in)

        # does dropout gives better performances? hmm..
        hidden_layer = tf.nn.dropout(hidden_layer,self.dropout_rate)
        positive_scores = tf.matmul(tf.expand_dims(hidden_layer,0), tf.gather(w_out,x,axis=1))
        negative_scores = tf.matmul(tf.expand_dims(hidden_layer,0), tf.gather(w_out,negative_items,axis=1))


        # rmse-like :
        #
        positive_errors = (1.0 - tf.sigmoid(positive_scores)) **2
        negative_errors = tf.sigmoid(negative_scores) **  2

        self.error = tf.reduce_mean(positive_errors) + 3.0*tf.reduce_mean(negative_errors)

        self.init = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
        self.sess.run(self.init)

        #self.reconstructed_x = activation(tf.nn.bias_add(tf.matmul(hidden_layer,w_out),b_out))
        
        #reconstructed_x = self.reconstructed_x


        self.pred_error = self.error

        
        self.pred_error_with_reg = self.pred_error + tf.contrib.layers.apply_regularization(regularizer, trainable_parameters)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.error,
            var_list = trainable_parameters,name='opt')

        self.init = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
        self.sess.run(self.init)
        self.saver = tf.train.Saver()

        self.row_input = tf.placeholder("float", [None,n_items],name='row')
        self.predicted = -tf.nn.bias_add(tf.matmul(activation(tf.nn.bias_add(tf.matmul(self.row_input,w_in),b_in)),w_out),b_out)



    def train_model(self,X,n_iter = 10,dropout_rate = 0.5,vad_data = None,**kwargs):
        '''Fit the model to the interaction matrix X
        Parameters
        ----------
        X : ??
        vad_data = same data type and same shape with X;
        '''
        self.n_users,self.n_items = X.shape
        n_users,n_items = X.shape
        vec = []
        for i in range(X.shape[0]):
            vec.append(X[i].nonzero()[1])
        negs = []
        for i in range(len(vec)*100):
            negs.append(np.random.choice(n_items,128,replace = True))

        self._init_model(n_users,n_items)
        elapsed_time = self._update(vec,negs,n_iter,dropout_rate,**kwargs)
    
    def _update(self,vec,negs,n_iter,dropout_rate,**kwargs):
        iter_state = range(n_iter)

        begin_time = time.time()
        for _ in iter_state:
            cost_on_iteration = self.run_epoch(vec,negs,dropout_rate)
            
            print(self.info_per_iter(_,kwargs.get('test_visible',None),kwargs.get('test_hidden',None),kwargs.get('k',10)))
        return time.time() - begin_time
    

    def run_epoch(self,vec,negs,dropout_rate):
        i = np.random.choice(len(vec),len(vec))
        p = np.random.choice(len(negs),len(vec))
        #print(i)
        #print(p)
        mp = []
        for _ in range(len(vec)):

            dbg,err,__ = self.sess.run([self.ret,self.pred_error_with_reg, self.optimizer],
                feed_dict = {
                self.items : vec[i[_]],
                #self.negative_items : negs[i[_]], 
                self.negative_items : negs[i[_]], 
                self.dropout_rate : dropout_rate})
            mp.append(err)
    def loss():
        pass

    def predict(self,X):
        return np.asarray(self.sess.run(self.predicted,feed_dict = {self.row_input : X}))


