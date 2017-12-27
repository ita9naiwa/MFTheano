"""
@author : hyunsung lee
@email : ita9naiwa@gmail.com
"""
# coding: utf-8

from .Recommender import Explicit_recommender

import time
import numpy as np
from numpy import linalg as LA
from joblib import Parallel, delayed
from scipy.sparse import *


class biasedMF_ALS(Explicit_recommender):

	model_name = 'biasedMF_ALS'

	def __init__(self, latent_dim = 10,batch_size = 1000,dtype='float32',
		n_jobs = -2,verbose = True,seed = 1234, **kwargs):
		
		super(biasedMF_ALS,self).__init__(dtype,verbose,seed,**kwargs)

		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.dtype = dtype
		self.n_jobs = n_jobs
		self.verbose = verbose
		self._parse_kwargs(**kwargs)

	def _parse_kwargs(self, **kwargs):
		self.lam_theta = float(kwargs.get('lambda_theta', 10.0))
		self.lam_beta = float(kwargs.get('lambda_beta', 10.0))
		self.lam_b = float(kwargs.get('lambda_b', 3.0))
		self.lam_c = float(kwargs.get('lambda_c', 3.0))
		self.init_std = float(kwargs.get('init_std',0.01))
		self.save_dir = str(kwargs.get('save_dir','./'))


	def _init_params(self,R):
		'''
		Initialize all the latent factors and biases
		'''
		n_users,n_items = R.shape
		self.theta = np.random.normal(0, self.init_std, size = (n_users,self.latent_dim)).astype(dtype=self.dtype)
		self.beta = np.random.normal(0, self.init_std, size = (n_items,self.latent_dim)).astype(dtype=self.dtype)
		self.bias_b 	= np.zeros(n_users, dtype=self.dtype)
		self.bias_c 	= np.zeros(n_items, dtype=self.dtype)
		self.mu 		= np.mean(R[R.nonzero()])

	def fit(self, R, vad_data=None,n_iters=10, **kwargs):
		'''Fit the model to the rating matirx R.

		Parameters
		----------
		R : scipy.sparse.csr_matrix, shape (n_users, n_items)
			Training rating matrix.
		vad_data: scipy.sparse.csr_matrix, shape (n_users, n_items)
			Validation rating data.
		**kwargs: dict
			Additional keywords to evaluation function call on validation data
		'''
		n_users, n_items = R.shape
		self._init_params(R)
		elapsed_time = self._update(R,vad_data,n_iters, **kwargs)
		print("elapsed time : %0.3fs"%elapsed_time)
		return self.validation_rmse(vad_data)

	def _update(self,R,vad_data,n_iters,**kwargs):
		RT = R.T.tocsr()
		begin_time = time.time()
		for itr in range(n_iters):
			self._update_mf(R,RT)
			if self.verbose:
				validation_rmse = self.validation_rmse(vad_data)
				if itr >= 3 and validation_rmse >= 0.8:
					break
				#train_loss = self.loss(R)
				self.iter_info_per_itr(itr,0,validation_rmse)
		return time.time() - begin_time

	def iter_info_per_itr(self,itr,train_loss,validation_rmse):
		info = str("[ Iteration #%d ]\t [ train loss %f ]\t [ validation rmse %f ]" % (itr,train_loss,validation_rmse))
		return info




	def _update_mf(self,R,RT):

		batch_size = self.batch_size

		self.beta 		= update_beta(self.theta,RT,
			self.lam_beta,self.bias_b,self.bias_c,self.mu,self.n_jobs,self.batch_size)	

		self.theta 	= update_theta(self.beta,R,self.lam_theta,
			self.bias_b,self.bias_c,self.mu,self.n_jobs,self.batch_size)

		
		self.bias_b = update_bias(self.theta,self.beta,self.bias_c,self.mu,self.lam_b,
			R,self.n_jobs,self.batch_size)
		
		self.bias_c		= update_bias(self.beta,self.theta,self.bias_b,self.mu,self.lam_c,
			RT,self.n_jobs,self.batch_size)

	def loss(self,R):
		R_pred = self.predict_matrix()
		diff = R[R.nonzero()] - R_pred[R.nonzero()]
		loss =  np.sum(np.square(diff))
		loss += self.lam_theta * np.sum(np.square(self.theta)) + self.lam_beta * np.sum(np.square(self.beta)) 
		loss += self.lam_b * np.sum(np.square(self.bias_b)) + self.lam_c * np.sum(np.square(self.bias_c))
		return loss

	def predict(self,u,i):
		return np.dot(self.theta[u],self.beta[i]) + self.bias_b[u] + self.bias_c[i] + self.mu

	def predict_matrix(self):
		q = np.dot(self.theta,self.beta.T)
		p = np.matrix(self.bias_b).T + np.dot(self.theta,self.beta.T)
		#print((p - q)[:2,:3])
		return np.asarray(np.matrix(self.bias_b).T + np.dot(self.theta,self.beta.T)+self.bias_c) + self.mu
	

	def validation_rmse(self,vad_data):
		rec_mat = self.predict_matrix()
		nz = vad_data.nonzero()
		vad_dense = np.asarray(vad_data.todense())
		error = rec_mat[nz] - vad_dense[nz]
		mse = np.mean(error * error)
		rmse = np.sqrt(mse)
		mae = np.abs(error).mean()
		#print("rmse : %f, mae : %f" % (rmse,mae))
		return rmse



def update_theta(beta,R,lam_theta,bias_b,bias_c,mu,n_jobs,batch_size = 1000):
	# m : n_users, n : n_items f : n_factors
	
	m, n = R.shape
	f = beta.shape[1]
	start_idx = np.arange(0,m,batch_size)
	end_idx = np.append(start_idx[1:],m)
	

	res = Parallel(n_jobs = n_jobs)(
		delayed(_solve_theta)(
			lo,hi,beta,R,lam_theta,bias_b,bias_c,mu,f)
		for lo,hi in zip(start_idx,end_idx))
	theta = np.vstack(res)
	return theta

def _solve_theta(lo,hi,beta,R,lam_theta,bias_b,bias_c,mu,f):
	lam_eye = lam_theta * np.eye(f,dtype = beta.dtype)
	theta_batch = np.empty((hi-lo,f),dtype = beta.dtype)
	#idx_u는 u가 본 item들의 인덱스들이니 i_1, i_2,...i_k겠지

	for ib, u in enumerate(range(lo,hi)):
		R_u, idx_u = get_row(R,u)
		beta_u = beta[idx_u]
		B_u = bias_c[idx_u] + bias_b[u] + mu
		lhs = beta_u.T.dot(beta_u) + lam_eye
		rhs = (R_u - B_u).dot(beta_u)
		theta_batch[ib] = LA.solve(lhs,rhs)
	return theta_batch

def update_beta(theta,RT,lam_beta,bias_b,bias_c,mu,n_jobs,batch_size = 1000):
	# m : n_users, n : n_items f : n_factors
	n,m = RT.shape
	f = theta.shape[1]
	assert theta.shape[0] == m
	start_idx = np.arange(0,n,batch_size)
	end_idx = np.append(start_idx[1:],n)	
	res = Parallel(n_jobs= n_jobs)(
		delayed(_solve_beta)(
			lo,hi,theta,RT,lam_beta,bias_b,bias_c,mu,f)
		for lo,hi in zip(start_idx,end_idx))
	beta = np.vstack(res)

	return beta


def _solve_beta(lo,hi,theta,RT,lam_beta,bias_b,bias_c,mu,f):
	beta_batch = np.empty((hi - lo,f),dtype = theta.dtype)
	lam_eye = lam_beta * np.eye(f,dtype = theta.dtype)

	for ib, i  in enumerate(range(lo,hi)):
		R_i,idx_i = get_row(RT,i)
		theta_i = theta[idx_i]
		B_i = bias_b[idx_i] + bias_c[i] + mu
		lhs = theta_i.T.dot(theta_i) + lam_eye
		rhs = (R_i - B_i).dot(theta_i)
		beta_batch[ib] = LA.solve(lhs,rhs)
	return beta_batch

def update_bias(theta,beta,bias_c,mu,lam_b,R,n_jobs,batch_size = 1000):
	#bias_b를 업데이트한다고 가정.
	#사실 bias_c를 업데이트 한다고 해도 아무런 차이 없음 ^^;;

	m = theta.shape[0] # n_users
	start_idx 	= np.arange(0,m,batch_size)
	end_idx 	= np.append(start_idx[1:],m)
	

	res = Parallel(n_jobs = n_jobs)(
		delayed(_solve_bias)(lo,hi,theta,beta,bias_c,mu,R,lam_b)
		for lo, hi in zip(start_idx,end_idx))
	bias_b = np.hstack(res)

	return bias_b

def _solve_bias(lo,hi,theta,beta,bias_c,mu,R,lam_b):

	bias_b_batch = np.empty(hi - lo, dtype = theta.dtype)
	for ib, u in enumerate(range(lo,hi)):
		R_u, idx_u = get_row(R,u)
		R_u_cardinal = np.float32(len(idx_u))
		rsd = (R_u-mu-bias_c[idx_u]).sum() - beta[idx_u].dot(theta[u]).sum(axis=0)
		bias_b_batch[ib] = rsd / (np.float32(R_u_cardinal) + lam_b)
	return bias_b_batch




# Utility functions #
def get_row(Y, i):
	'''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
	non-zero values in i_th row'''
	lo, hi = Y.indptr[i], Y.indptr[i + 1]
	return Y.data[lo:hi], Y.indices[lo:hi]


if __name__ == "__main__":
	#debug code
	model = biasedMF_ALS(latent_dim = 20)
	R = csr_matrix(np.array([[1,2],[3,4]]))
	model.fit(R,R,3)
	print(model.predict_matrix())