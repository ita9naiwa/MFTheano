from .Recommender import Implicit_recommender
from tqdm import tqdm
import random
import os
import sys
import time
import numpy as np



class BPR(Implicit_recommender):
	def __init__(self,dtype = 'float32',verbose=True,seed=None,**kwargs):
		np.seterr(divide='ignore', invalid='ignore')
		super(BPR,self).__init__(dtype,verbose,seed,**kwargs)
		self.model_name = 'BPR'
		self.seed = seed
		self._parse_kwargs(**kwargs)

	def _parse_kwargs(self,**kwargs):
		self.latent_dim = int(kwargs.get('latent_dim',20))
		self.learning_rate = float(kwargs.get('leraning_rate',0.05))
		self.lamb_theta = float(kwargs.get('lambda_total',0.1))
		self.lamb_beta = float(kwargs.get('lambda_total',0.1))
		self.bold_driver = str(kwargs.get('activation','sigmoid'))
		self.use_item_bias = bool(kwargs.get('use_item_bias',False))
		self.init_std = float(kwargs.get('init_std',0.01))
		self.sample_proportion = int(kwargs.get('sample_proportion',5))



	def _init_model(self,n_users,n_items):
		self.theta = np.random.normal(0, self.init_std, size = (n_users,self.latent_dim)).astype(dtype=self.dtype)
		self.beta = np.random.normal(0, self.init_std, size = (n_items,self.latent_dim)).astype(dtype=self.dtype)
		self.item_bias = np.zeros(n_items, dtype=self.dtype)

	def train_model(self,X,n_iters = 10,vad_data = None, **kwargs):
		n_users,n_items = X.shape
		
		self._init_model(n_users,n_items)
		elapsed_time = self._update(X,vad_data,n_iters,**kwargs)

	def _update(self,X,vad_data,n_iters = 15,n_sample = 50000,**kwargs):
		n_users,n_items = X.shape

		zp = X.nonzero()
		zp = [zp[0].tolist(),zp[1].tolist()]
		interactions = list(zip(*zp))
		set_interactions =set(interactions)
		interactions = list(interactions)
		if True == self.verbose:
			iter_state = tqdm(range(n_iters))
		else:
			iter_state = range(n_iters)
		
		begin_time = time.time()
		for _ in iter_state:
			cost_on_iteration = self.run_epoch(n_users,n_items,
				interactions,set_interactions, n_sample)

			if self.verbose:
				print(self.iter_info_per_itr())

		return time.time() - begin_time

	def run_epoch(self,n_users,n_items,
		interactions,set_interactions, n_sample):

		ret = self._input_interaction_builder(n_users,n_items,interactions,set_interactions,
			n_sample,self.sample_proportion)
		for u,i,j in ret:
			self._grad_single_point(u,i,j)

	def predict(self,X):
		pred_matrix = np.dot(self.theta,self.beta.T)
		pred_matrix[X.nonzero()] = -500000.0
		pred_matrix = -pred_matrix
		print(pred_matrix.shape)
		return pred_matrix

	def predict_point(self,u,i):
		ret = np.dot(self.theta[u],self.beta[i])
		return ret

	def loss(self,X):
		return 0
	def predict_matrix(self,X):
		return 0



	def _grad_single_point(self,u,i,j):
		beta_diff = self.beta[i] - self.beta[j]
		l_t = self.lamb_theta
		l_b = self.lamb_beta
		x_uij = np.dot(self.theta[u],beta_diff)
		def sigmoid(x):
			exp = min(10000.0,np.exp(-x))
			return exp / (1.0 + exp)
			

		grad = sigmoid(x_uij) * self.learning_rate
		self.theta[u] += grad * (beta_diff - l_t * self.theta[u])
		self.beta[i] += grad * (self.theta[u] - l_b * self.beta[i])
		self.beta[j] -= grad * (self.theta[u] + l_b * self.beta[j])

	def _input_interaction_builder(self,n_users,n_items,
		interactions,set_interactions,n_sample = 5000,
		sample_proportion = 8,
		np_parallize = 1024):

		sampled_poss = random.sample(interactions,np.min([len(interactions),n_sample]))
		ret = []

		for u,i in sampled_poss:
			js = np.random.choice(n_items,np_parallize)
			cnt = 0
			for j in js:
				if (u,j) not in set_interactions:
					ret.append((u,i,j))
					cnt += 1
				if cnt >= sample_proportion:
					break
		np.random.shuffle(ret)
		return ret
