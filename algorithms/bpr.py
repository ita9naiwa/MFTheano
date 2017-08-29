
import time
import pickle
import numpy as np
import pandas as pd
from random import shuffle
import theano
import theano.tensor as T

import sys
#it is not good I think...
import optimizers


def randinit(shape,scaleFactor = 1.0/0.6 ,dtype = theano.config.floatX):
	scale = np.sqrt(np.sum(shape))
	return np.random.normal(loc=0.0, scale=3.0/scale, size=shape).astype(dtype)

class BPR_MF(object):
	def __init__(self,unique_users,unique_items,latent_dim = 16,learning_rate = 0.001,reg = 0.03):

		super(BPR_MF, self).__init__()
		self.save_param = False
		
		self.latent_dim = latent_dim
		self.num_users = unique_users
		self.num_items = unique_items
		self.lr = learning_rate
		#item matrix
		self.I = theano.shared(value = randinit((self.num_items,latent_dim)), name = 'I', borrow = 'True')
		#user matrix
		self.U = theano.shared(value = randinit((self.num_users,latent_dim)),name='U',borrow='True')
		
		# input
		# indices for user,item, negative_item
		self.users = T.iscalar()
		self.items = T.iscalar()
		self.items_neg = T.iscalar()
		self.pred = T.dot(self.U[self.users] , self.I[self.items])
		self.pred_neg = T.dot(self.U[self.users] , self.I[self.items_neg])

		
		self.BPR_loss = -T.log(T.nnet.sigmoid(self.pred - self.pred_neg)) + reg * (T.sum(self.U ** 2) + T.sum(self.I ** 2))



		#loss and update function
		self.loss = self.BPR_loss

		self.update = optimizers.sgd(self.loss,[self.U,self.I], learning_rate = self.lr)
		self.predict = theano.function([self.users,self.items],[self.pred,self.items])
		self.train_MF = theano.function([self.users,self.items,self.items_neg],[self.loss],updates =self.update)

	def reset_lr(self,q = None,p = None):
		if q != None:
			self.lr = self.lr * q
			self.update = optimizers.sgd(self.loss,[self.U,self.I], learning_rate = self.lr)
			self.train_MF = theano.function([self.users,self.items,self.items_neg],[self.loss],updates =self.update)
		elif p != None:
			self.lr = p
			self.update = optimizers.sgd(self.loss,[self.U,self.I], learning_rate = self.lr)
			self.train_MF = theano.function([self.users,self.items,self.items_neg],[self.loss],updates =self.update)

		
	def train_mf(self,users,items,neg_items):

		losses = []
		for idx in range(len(users)):
			#if ((idx+1) % 10000) == 0:
			#	print(idx+1)
			[loss] = self.train_MF(users[idx],items[idx],neg_items[idx])
			losses.append(loss)

		return np.mean(losses)

	def save(self):
	    U,V = self.getUV(fname)
	    np.savez(fname, U=U,V=V)
	def get_params(self):
	    U = self.U.eval()
	    V = self.I.eval()
	    return U,V

