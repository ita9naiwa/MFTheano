# coding: utf-8

from .Recommender import Explicit_recommender

import time
import sys
import random
import numpy as np
from scipy.sparse import csr_matrix,lil_matrix

class biasedMF_SGD(Explicit_recommender):

	model_name = 'biasedMF_SGD'
	
	def __init__(self, latent_dim=10,dtype='float32',verbose=True,seed = 1234,**kwargs):
		
		super(biasedMF_SGD,self).__init__(dtype,verbose,seed,**kwargs)

		random.seed(1541)
		self._parse_kwargs(**kwargs)
		self.latent_dim = latent_dim  # number of latent_dim.
		self.verbose = verbose
		self.dtype = dtype
		# Init model parameters

	def _parse_kwargs(self,**kwargs):
		self.reg_mat = float(kwargs.get('reg_mat', 0.05))  # regularization parameters
		self.reg_bias = float(kwargs.get('reg_bias', 0.05))
		self.lr = lr = float(kwargs.get('learning_rate',0.03))
		self.use_bold_driver = bool(kwargs.get('use_bold_driver',True))
		self.init_std = float(kwargs.get('init_std',0.01))

	def _init_params(self,R):
		n_users,n_items = R.shape
		self.theta = np.random.normal(0, self.init_std, size = (n_users,self.latent_dim)).astype(dtype=self.dtype)
		self.beta = np.random.normal(0, self.init_std, size = (n_items,self.latent_dim)).astype(dtype=self.dtype)
		self.bias_b = np.zeros(shape = n_users).astype(self.dtype)
		self.bias_c = np.zeros(shape = n_items).astype(self.dtype)
		self.mu   = R[R.nonzero()].mean()

	def fit(self,R,vad_data,n_iter):
		loss_pre = sys.float_info.max
		self._init_params(R)
		train_nz_tuples = list(zip(*R.nonzero()))
		nonzeros = R.nnz
		R = lil_matrix(R)
		test_nz = vad_data.nonzero()

		auc_prev = 0.0
		prev_time = time.time()
		for itr in range(n_iter):
			start = time.time()
			# Each training epoch
			diffs = []
			for s in range(nonzeros):
				u,i = random.choice(train_nz_tuples)
				r_pred = self.predict(u,i)
				r = R[u,i]
				diff = r_pred - r
				diffs.append(diff)
				self.update_ui(diff, u, i)
			if self.use_bold_driver:
				loss_curr = self.loss(R)
				loss_pre = self._bold_driver(itr, start, loss_pre,loss_curr)
				loss_pre = loss_curr
			if True == self.verbose:
				train_loss = self.loss(R)
				validation_rmse = self.validation_rmse(itr,vad_data,test_nz)
				self.iter_info_per_itr(itr,train_loss,validation_rmse)
		elapsed_time = time.time() - prev_time
		print("elapsed time : %0.3fs"%elapsed_time)

	def iter_info_per_itr(self,itr,train_loss,validation_rmse):
		print("[ Iteration #%d ]\t [ train loss %f ]\t [ validation rmse %f ]" % (itr,train_loss,validation_rmse))


	def update_ui(self,diff,u,i):
		for f in range(self.latent_dim):
			self.theta[u,f] -= self.lr * (diff * self.beta[i,f] + self.reg_mat * self.theta[u,f])
			self.beta[i,f] -= self.lr * (diff * self.theta[u,f] + self.reg_mat * self.beta[i,f])
		self.bias_b[u] -= self.lr * (diff + self.reg_bias * self.bias_b[u])
		self.bias_c[i] -= self.lr * (diff + self.reg_bias * self.bias_c[i])
	
	def _bold_driver(self, itr, start, loss_pre,loss_curr):
		if loss_pre >= loss_curr:
			symbol = "-"
			self.lr *= 1.05
			self.lr = np.min([self.lr,0.05])
		else:
			symbol = "+"
			self.lr *= 0.5
			self.lr = np.max([self.lr,0.001])
		#print("current lr : %f" % self.lr)
		#print("Iter={} [{}]\t [{}]loss: {} [{}]\n".format(itr, start1 - start, symbol, loss_curr, time.time() - start1))

	def loss(self,R):
		n_users,n_items = R.shape
		L = self.reg_mat * (np.sum(np.square(self.theta)) + np.sum(np.square(self.beta))) + self.reg_bias * (np.power(self.bias_c,2).sum() + np.power(self.bias_b,2).sum())
		for u in range(n_users):
			l = 0
			for i in R.getrowview(u).rows[0]:
				pred = self.predict(u, i)
				l += np.power(R[u, i] - pred, 2)
			L += l
		return L

	def predict(self, u, i):
		return np.dot(self.theta[u], self.beta[i]) + self.bias_b[u] + self.bias_c[i] + self.mu
	def validation_rmse(self,itr,vad_data,test_nz):
		diffs = []

		for u,i in zip(*test_nz):
			pred = self.predict(u,i)
			true = vad_data[u,i]
			diffs.append(pred - true)
		
		#mae = np.abs(diffs).mean()
		mse = np.power(np.array(diffs),2).mean()
		rmse = np.sqrt(mse)
		return rmse
	
	def predict_matrix(self):
		q = np.dot(self.theta,self.beta.T)
		p = np.matrix(self.bias_b).T + np.dot(self.theta,self.beta.T)
		#print((p - q)[:2,:3])
		return np.asarray(np.matrix(self.bias_b).T + np.dot(self.theta,self.beta.T)+self.bias_c) + self.mu
