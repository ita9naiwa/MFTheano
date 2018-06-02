from .Recommender import ImplicitRecommender
from tqdm import tqdm
import random
import os
import sys
import time
import numpy as np



class BPR(ImplicitRecommender):
	def __init__(self,dtype = 'float32',verbose=True,seed=None,**kwargs):
		np.seterr(divide='ignore', invalid='ignore')
		super(BPR,self).__init__(dtype, verbose, seed, **kwargs)
		self.model_name = 'BPR'
		self.seed = seed
		self._parse_kwargs(**kwargs)

	def _parse_kwargs(self,**kwargs):
		self.latent_dim = int(kwargs.get('latent_dim', 20))
		self.learning_rate = float(kwargs.get('leraning_rate', 0.05))
		self.lamb_theta = float(kwargs.get('lambda_total', 0.1))
		self.lamb_beta = float(kwargs.get('lambda_total', 0.1))
		self.use_item_bias = bool(kwargs.get('use_item_bias', False))
		self.init_std = float(kwargs.get('init_std', 0.01))
		self.sample_proportion = int(kwargs.get('sample_proportion', 5))

	def _init_model(self,n_users,n_items):
		self.theta = np.random.normal(0, self.init_std,
			size=(n_users, self.latent_dim)).astype(self.dtype)
		self.beta = np.random.normal(0, self.init_std,
			size=(n_items, self.latent_dim)).astype(self.dtype)

		self.item_bias = np.zeros(n_items, dtype=self.dtype)

	def train_model(self,X,n_iter = 10,vad_data = None, **kwargs):
		n_users, n_items = X.shape
		self._init_model(n_users, n_items)
		_X = X.tocoo()
		elapsed_time = self._update(_X,vad_data,n_iter,**kwargs)

	def _update(self,X,vad_data,n_iter = 15,n_sample = 50000,**kwargs):
		n_users, n_items = X.shape
		row, col, val = X.row, X.col, X.data

		pos_interactions = []
		neg_interactions = []

		for u, i, y  in zip(row, col, val):
			if y > 0:
				pos_interactions.append((u, i))
			else:
				neg_interactions.append((u, i))

		set_pos_interactions = set(pos_interactions)
		set_neg_interacitons = set(neg_interactions)

		begin_time = time.time()

		for _ in tqdm(range(n_iter),disable=not self.verbose):
			cost_on_iteration = self.run_epoch(n_users, n_items,
				pos_interactions, set_pos_interactions,
				neg_interactions, set_neg_interacitons,
				n_sample)

		return time.time() - begin_time

	def run_epoch(self, n_users, n_items,
		pos_interactions, set_pos_interactions,
		neg_interactions, set_neg_interacitons,
		n_sample):

		ret = self._input_interaction_builder(n_users, n_items,
			pos_interactions,set_pos_interactions,
			neg_interactions, set_neg_interacitons,
			n_sample, self.sample_proportion)

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
		"""not implemented yet"""
		return 0

	def predict_matrix(self,X):
		"""not implemented yet"""
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

	def _input_interaction_builder(self, n_users, n_items,
	    pos_interactions, set_pos_interactions,
	    neg_interactions, set_neg_interacitons,
	    n_sample=5000,
	    sample_proportion=2,
	    np_parallize=1024):
	    sampled_pos = random.sample(pos_interactions, min(len(pos_interactions), n_sample))
	    sampled_neg = random.sample(neg_interactions, min(len(neg_interactions), n_sample))
	    ret = []

	    for u, i in sampled_pos:
	        js = np.random.choice(n_items, np_parallize)
	        cnt = 0
	        for j in js:
	            if (u,j) not in set_pos_interactions:
	                ret.append((u, i, j))
	                cnt += 1
	            if cnt >= sample_proportion:
	                break

	    for u, i in sampled_neg:
	        js = np.random.choice(n_items, np_parallize)
	        cnt = 0
	        for j in js:
	            if (u,j) not in set_neg_interacitons:
	                ret.append((u, j, i))
	                cnt += 1
	            if cnt >= sample_proportion:
	                break
	    np.random.shuffle(ret)
	    return ret
