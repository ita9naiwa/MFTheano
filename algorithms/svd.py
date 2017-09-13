
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

class SVD(object):
	def __init__(self,unique_users,unique_items,latent_dim = 16,learning_rate = 0.01,reg = 0.01):

		super(SVD, self).__init__()
		self.save_param = False
		
		self.latent_dim = latent_dim
		self.num_users = unique_users
		self.num_items = unique_items
		self.lr = learning_rate
		#user, item matrix

		self.U = theano.shared(value = randinit((self.num_users,latent_dim)),name='U',borrow='True')
		self.I = theano.shared(value = randinit((self.num_items,latent_dim)), name = 'I', borrow = 'True')
		
		#user bias, item bias
		self.bias_u = theano.shared(value = randinit((self.num_users,)), name= 'bias_u',borrow='True')
		self.bias_i = theano.shared(value = randinit((self.num_items,)), name= 'bias_i',borrow='True')

		# input
		# indices for user,item, negative_item
		self.user = T.iscalar()
		self.item = T.iscalar()
		self.rating = T.iscalar()
		self.users = T.ivector()
		self.items = T.ivector()

		self.pred = T.dot(self.U[self.user] , self.I[self.item].T) + self.bias_u[self.user] + self.bias_i[self.item]
		self.pred_list = T.dot(self.U[self.users] ,self.I[self.items].T) + self.bias_u[self.users] + self.bias_i[self.items]

		self.loss = (self.pred - self.rating) ** 2 + reg * (T.sum(self.U[self.user] ** 2) + T.sum(self.I[self.item] ** 2) + T.sum(self.bias_u**2) + T.sum(self.bias_i))

		self.update = optimizers.sgd(self.loss,[self.U,self.I], learning_rate = self.lr)
		self.predict = theano.function([self.user,self.item],[self.pred,self.item])
		self.train_MF = theano.function([self.user,self.item,self.rating],[self.loss],updates =self.update)
		
		self.predict_list = theano.function([self.users,self.items],[self.pred_list])
	"""
	def reset_lr(self,q = None,p = None):
		if q != None:
			self.lr = self.lr * q
			self.update = optimizers.sgd(self.loss,[self.U,self.I], learning_rate = self.lr)
			self.train_MF = theano.function([self.user,self.item,self.item_neg],[self.loss],updates =self.update)
		elif p != None:
			self.lr = p
			self.update = optimizers.sgd(self.loss,[self.U,self.I], learning_rate = self.lr)
			self.train_MF = theano.function([self.user,self.item,self.item_neg],[self.loss],updates =self.update)
	"""
	def RMSE(self,users,items,ratings):
		[pred_ratings] = self.predict_list(users,items)
		return np.sqrt(np.mean((pred_ratings - ratings)**2))


	def train(self,users,items,ratings):
		losses = []
		for idx in range(len(users)):
			[loss] = self.train_MF(users[idx],items[idx],ratings[idx])
			losses.append(loss)

		return np.mean(losses)



