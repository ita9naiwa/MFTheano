import numpy as np
import pandas as pd
from random import shuffle
import theano
import theano.tensor as T
import optimizers

def randinit(shape,scaleFactor = 1.0/0.6 ,dtype = theano.config.floatX):
	scale = np.sqrt(np.sum(shape))
	return np.random.normal(loc=0.0, scale=3.0/scale, size=shape).astype(dtype)

class NCF(object):
	def __init__(self,num_users,num_items,latent_dim = 16,learning_rate = 0.001,lamb= 0.001):
		super(NCF, self).__init__()

		self.latent_dim = latent_dim
		self.num_users = num_users
		self.num_items = num_items


		#input
		self.users = T.ivector()
		self.items = T.ivector()
		self.ratings = T.ivector()

		#latent features
		self.user_embeddings_1 = theano.shared(value = randinit((self.num_users,latent_dim)) , name = 'U_1', borrow = 'True')
		self.user_embeddings_2 = theano.shared(value = randinit((self.num_users,latent_dim)) , name = 'U_1', borrow = 'True')
		self.item_embeddings_1 = theano.shared(value = randinit((self.num_items,latent_dim)) , name = 'I_1', borrow = 'True')
		self.item_embeddings_2 = theano.shared(value = randinit((self.num_items,latent_dim)) , name = 'I_2', borrow = 'True')

		"""
		NCF have two interaction modellings.
			(1)  weighted linear combination.
			(2)  fully conneceted multi-layer perceoptron with  user vector and item vectors concatenated-layer 

		"""
		# (1) weighted linear combination(generalized matrix factorization)
		#(num_input, latent_dim) size matrix
		# (num_input,latent_dim)
		out_gmf = user_embeddings_1[self.users] * ite_embeddings_1[self.items]

		# (1) fully connected multi-layer perceptron with user vector and item vectors.
		#(num_input, latent_dim) size matrix
		#(num_input,latent_dim)
		l1 = T.concatenate([self.item_embeddings_1,self.item_embeddings_2],axis = 1)
		layer_1 = theano.shared(value = randinit((2*latent_dim,latent_dim)),name='L1',borrow = True)
		bias_1 = theano.shared(value = randinit((latent_dim,)), borrow = True)
		l2 = T.sigmoid(T.dot(l1,layer_1)) + bias_1
		layer_2 = theano.shared(value = randinit((latent_dim,latent_dim)),name='L2',borrow = True)
		bias_1 = theano.shared(value = randinit((latent_dim,)), borrow = True)
		out_mlp = T.sigmoid(T.dot(l2,layer_2)) + bias_2

		output_layer = theano.shared(value = randinit((2*latent_dim,1)), name = 'output',borrow = True)

		preds = T.sigmoid(T.dot(T.concatenate([out_gmf,out_mlp],axis = 1), output_layer))

		regs =  T.sum(user_embeddings_1 **2 +user_embeddings_2 **2) + T.sum(item_embeddings_1 **2 + item_embeddings_2 **2)
		regs += T.sum(layer_1**2) + T.sum(layer_2**2)
		regs += T.sum(bias_1**2) + T.sum(bias_2**2)
		regs += T.sum(output_layer**2)
		
		error = T.mean(T.nnet.nnet.binary_crossentropy(preds,self.ratings)) + lamb * regs

	def train(self,epochs = 100,batch_size = 256):
		updates = optimizers.adam(error,[self.P,self.Q,self.b_i,self.b_u,self.g_b], learning_rate = learning_rate)
