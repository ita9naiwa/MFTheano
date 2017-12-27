# abc means abstract base class
# force base class to be abstract, check this http://bluese05.tistory.com/61

from abc import ABC,abstractmethod,abstractclassmethod
import numpy as np 
from scipy.sparse import *
import random
import logging

def print_arbitrary(dic):
	ret = ""
	for name,var in dic.items():
		ret += "[%s : %s]\t" % (name,str(var))
	print(ret)

class Recommender(ABC):
	
	"""
	Abstract Class for Recommender
	Every recommender inherits this class (will, hope so)

	I'll expect every recommender, at least have these
	variables, methods, and others.
	
	"""

	#Class static variable이네.
	model_name = 'mymodel'


	def desc(self):
		print(self.model_name)


	def __init__(self,dtype = 'float32',verbose = True,seed = 1234,**kwargs):
		self.model_name = 'Recommender'
		self.logger = logging.getLogger(self.model_name)
		random.seed(seed)
		self.dtype = dtype
		self.verbose = verbose
		self._parse_kwargs(**kwargs)

	@abstractmethod
	def _parse_kwargs(self,**kwargs):
		self.a = 3
	
	@abstractmethod
	def _init_model(self,):
		pass

	@abstractmethod
	def train_model(self,):
		pass

	@abstractmethod
	def predict(self,):
		pass

	@abstractmethod
	def predict_matrix(self,):
		pass

	@abstractmethod
	def loss(self,):
		pass

	@abstractmethod
	def iter_info_per_itr(self,):
		pass






class Implicit_recommender(Recommender):
	
	model_name = 'implicit_recommender'


	def __init__(self, dtype = 'float32', verbose = True, seed =1234,**kwargs):
		super(Implicit_recommender, self).__init__(dtype,verbose,seed,**kwargs)
	
	
	def precision_recall_at_k(self,train_data,test_data,k):
		X_pred = self.predict(train_data)
		idx = np.argpartition(X_pred,k,axis = 1)
		X_pred_binary = np.zeros_like(test_data,dtype = bool)
		X_pred_binary[np.arange(train_data.shape[0])[:, np.newaxis], idx[:, :k]] = True
		tmp = (np.logical_and(test_data, X_pred_binary).sum(axis=1)).astype(
			np.float32)
		return np.nanmean(tmp/k),np.nanmean(tmp/test_data.sum(axis=1))
		

	
	def recall_at_ks(self,train_data,test_data,k):
		pass
	

	#@abstractmethod
	#def validation_ndcg_at_k(self,):
	#	pass





class Explicit_recommender(Recommender):
	
	model_name = 'explicit_recommender'

	def __init__(self, dtype = 'float32', verbose = True, seed =1234,**kwargs):
		super(Explicit_recommender, self).__init__(dtype,verbose,seed,**kwargs)


	@abstractmethod
	def validation_rmse(self,):
		pass





