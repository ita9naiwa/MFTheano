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
	def __init__(self,num_users,num_items,latent_dim = 16,learning_rate = 0.001,lambda_matrix = 0.03,lambda_bias = 0.001):
		super(NCF, self).__init__()

		self.latent_dim = latent_dim
		self.num_users = num_users
		self.num_items = num_items


		#input
		self.users = T.ivector()
		self.items = T.ivector()
		self.ratings = T.ivector()

		#latent features
		self.user_embeddings_1 = theano.shared(value = randinit((self.num_users,latent_dim) , name = 'U_1', borrow = 'True')
		self.user_embeddings_2 = theano.shared(value = randinit((self.num_users,latent_dim) , name = 'U_2', borrow = 'True')
		self.item_embeddings_1 = theano.shared(value = randinit((self.num_items,latent_dim) , name = 'I_1', borrow = 'True')
		self.item_embeddings_2 = theano.shared(value = randinit((self.num_items,latent_dim) , name = 'I_2', borrow = 'True')

		"""
		
		NCF have two interaction modellings.
			(1)  weighted linear combination.
			(2)  fully conneceted multi-layer perceoptron with  user vector and item vectors concatenated-layer 

		"""
		# (1) weighted linear combination.
		#(num_input, latent_dim) size matrix
		H = theano.shared(value = randinit((latent_dim, )),
			name = 'H',
			borrow='False')

		# (num_input,latent_dim)
		# epmirically adding H is not helpful.
		out_1 = T.sum((H * (user_embeddings_1 * ite_embeddings_1)),axis= 1)

		# (2)



		
		



		
	def train(self,epochs = 100,batch_size = 256):
		updates = optimizers.adam(self.loss,[self.P,self.Q,self.b_i,self.b_u,self.g_b], learning_rate = learning_rate)
