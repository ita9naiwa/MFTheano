from .Recommender import Implicit_recommender

from tqdm import tqdm

import os
import sys
import time
import numpy as np

class BPR(Implicit_recommender):
	def __init__(self,dtype = 'float32',verbose=True,seed=None,**kwargs):
		super(BPR,self).__init__(dtype,verbose,seed,**kwargs)
		self.model_name = 'BPR'
		self.seed = seed
		self.latent_dim = latent_dim
		self._parse_kwargs(**kwargs)

	def _parse_kwargs(self,**kwargs):
		self.latent_dim = int(kwargs.get('latent_dim',10))
		self.learning_rate = float(kwargs.get('leraning_rate',0.05))
		self.lamb_theta = float(kwargs.get('lambda_total',0.05))
		self.lamb_beta = float(kwargs.get('lambda_total',0.05))
		self.bold_driver = str(kwargs.get('activation','sigmoid'))
		self.use_item_bias = bool(kwargs.get('use_item_bias'),False)
		self.init_std = float(kwargs.get('init_std',0.001))


	def _init_model(self,n_users,n_items):
		self.theta = np.random.normal(0, self.init_std, size = (n_users,self.latent_dim)).astype(dtype=self.dtype)
		self.beta = np.random.normal(0, self.init_std, size = (n_items,self.latent_dim)).astype(dtype=self.dtype)
		self.item_bias = np.zeros(n_items, dtype=self.dtype)

	def train_model(self,X,n_iters,vad_data = None, **kwargs):
		n_users,n_items = X.shape
		
		self._init_model(n_users,n_items):
		elapsed_time = self._update(X,vad_data,n_iters,batch_size,**kwargs)

	def _update(X,vad_data,n_iters = 15,n_sample = 50000,**kwargs):

		if True == self.verbose:
			iter_state = tqdm(range(n_iters))
		else:
			iter_state = range(n_iters)

		begin_time = time.time()

		for _ in iter_state:
			cost_on_iteration = self.run_epoch(X,n_sample)
			if self.verbose:
				print(self.iter_info_per_itr())

		return time.time() - begin_time

	def run_epoch(self,X,n_samples):
		


	def _grad_single_point(u,i,j):

		theta_u = self.theta[u]
		beta_i = self.beta[i]
		beta_j = self.beta[j]

		beta_diff = beta_i - beta_j
		x_uij = np.dot(theta[u],beta_diff)
		def sigmoid(x):
			return np.min([1.0,np.exp(-x) / (1.0 + np.exp(-x))])

		grad = sigmoid(x_uij) * alpha
		theta[u] += grad * beta_diff + lam_theta * theta_u
		beta_i += grad * theta_u + lam_beta * beta_i
		beta_j += -grad* theta_u + lam_beta * beta_j

	def _input_interaction_builder(self,
		interactions,n_users,n_items,
		n_pos_samples = 5000,n_negative_samples_per_pos = 8,
		np_parallize = 64):

		sampled_poss = random.sample(interactions,n_pos_samples)
		set_interactions = set(interactions)
		

		ret = []
		for u,i in sampled_poss:
			js = np.random.choice(n_items,np_parallize)
			cnt = 0
			for j in js:
				if (u,j) not in set_interactions:
					ret.append((u,i,j))
					cnt += 1
					if cnt >= n_negative_samples_per_pos:
						break


		np.random.shuffle(ret)
		return ret
