"""
@author : Hyunsung lee
@email : ita9naiwa@gmail.com
"""

from .Recommender import ImplicitRecommender

import json
import tqdm
import logging
from  gensim.models import Word2Vec
from scipy.sparse import csr_matrix,lil_matrix
import os,sys
import time
import numpy as np 


class Prod2vec(ImplicitRecommender):
	def __init__(self,dtype = 'float32',verbose = True,seed=None,**kwargs):
		super(Prod2vec,self).__init__(dtype,verbose,seed,**kwargs)
		self.model_name = 'Prod2vec'
		self.seed = seed
		# items which appears less than min_count should be erased first
		self.min_count = 0
		self._parse_kwargs(**kwargs)

	def _parse_kwargs(self,**kwargs):
		self.latent_dim = int(kwargs.get('latent_dim',100))
		self.n_jobs = int(kwargs.get('n_jobs',8))
		self.learning_rate = float(kwargs.get('learning_rate',0.05))
		self.n_negatives = int(kwargs.get('n_negatives',10))

	def _init_model(self):
		# gensim word2vec initializes model as training begins
		pass
	def train_model(self,X,n_iter,**kwargs):
		self.n_users,self.n_items = X.shape
		begin_time = time.time()
		dataset, _ = self.matrix_to_lil(X,to_str = True)

		print("training begins")
		self.model = Word2Vec(dataset,
			size = self.latent_dim,
			seed = self.seed,
			alpha = self.learning_rate,
			workers = self.n_jobs,
			negative = self.n_negatives,
			iter = n_iter,
			window = 123456789,
			min_count = 0,
			sg = 1)
		elapsed_time = time.time() - begin_time

	def info_per_iter(Self):
		ret = "nothing to say for now..."
		return ret
	
	def predict(self,X):
		test_rows, _ = self.matrix_to_lil(X,to_str = True,
			ignore_empty_row = False,test = True)

		ret = np.zeros(X.shape,dtype = self.dtype)
		for i,row in enumerate(test_rows):
			if len(row) > 0:
				for item_id,sim in self.model.wv.most_similar(row,topn=self.n_items):
					ret[i,int(item_id)] = sim
			else:
				pass
		print(ret.shape)

		return -ret









	def loss(self):
		pass

	def matrix_to_lil(self,X,to_str = True,ignore_empty_row = True,test = False):
		lil_X = X.tolil()
		dataset = []
		ignored_rows = []
		for i in range(X.shape[0]):
			row = lil_X.getrow(i).nonzero()[1].tolist()
			
			if True == to_str:
				if True == test:
					dataset.append([str(x) for x in row if str(x) in self.model.wv.vocab])
				else:
					dataset.append([str(x) for x in row if str(x)])
			else:
				dataset.append(row)

		return dataset, ignored_rows
			
