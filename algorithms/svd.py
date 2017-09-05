import numpy as np
import pandas as pd
from random import shuffle
import theano
import theano.tensor as T
import optimizers

class SVD(object):
	def __init__(self,users_train,users_test,items_train,items_test,ratings_train,ratings_test,latent_dim = 16,learning_rate = 0.001,lambda_matrix = 0.03,lambda_bias = 0.001):
		super(SVD, self).__init__()


		self.latent_dim = latent_dim
		self.num_users = self.num_users
		self.num_items = self.num_items

		


		#input
		self.users = T.ivector()
		self.items = T.ivector()
		self.ratings = T.ivector()

		#user matrix
		self.P = theano.shared(value = np.random.normal(loc=0.0, scale=0.1, size=(self.num_users,latent_dim)).astype(theano.config.floatX)
				, name = 'P', borrow = 'True')

		#item matrix
		self.Q = theano.shared(value = np.random.normal(loc=0.0, scale=0.1, size=(self.num_items,latent_dim)).astype(theano.config.floatX)
				, name = 'Q', borrow = 'True')

		#vector biases
		self.b_i = theano.shared(value = np.random.normal(loc=0.0, scale=0.05, size=(self.num_items,)).astype(theano.config.floatX),name='item bias',borrow = True)
		self.b_u = theano.shared(value = np.random.normal(loc=0.0, scale=0.05, size=(self.num_users,)).astype(theano.config.floatX),name='user bias',borrow = True)

		#scalar biases
		self.g_b = theano.shared(value = np.random.normal(loc=0.0, scale=0.05, size=(1,)).astype(theano.config.floatX)[0],			name='gloabal bias',borrow = True)



		#variables in the computational graph
		#[batch size by 1]
		self.pred_ratings = ((T.sum(self.P[self.users] * self.Q[self.items],axis = 1) + self.b_u[self.users] + self.b_i[self.items]) + self.g_b)

		#[batch_size by 1]
		self.error = (self.pred_ratings - self.ratings.T)

		#scalar
		self.rmse = T.sqrt(T.mean(self.error ** 2))

		#scalar
		self.loss = T.sum(self.error * self.error) + lambda_matrix * (T.sum(self.P[self.users] ** 2) + T.sum(self.Q[self.items] ** 2)) + lambda_bias * (T.sum(self.b_i[self.items] ** 2) + T.sum(self.b_u[self.users] **2))


		
		#rating prediction given user and item
		self.predict = theano.function([self.users,self.items],[self.pred_ratings])



		
	def train(self,epochs = 100,batch_size = 256):
		updates = optimizers.adam(self.loss,[self.P,self.Q,self.b_i,self.b_u,self.g_b], learning_rate = learning_rate)
		train_model = theano.function([self.users,self.items,self.ratings],[self.loss],updates = updates)
		test_model = theano.function([self.users,self.items,self.ratings],[self.rmse,self.error])
		
		for epoch in range(epochs):
			losses = []

			for batch in range(0,len(self.users_train),batch_size):
				[loss] = train_model(self.users_train[batch:batch + batch_size],self.items_train[batch:batch + batch_size],self.ratings_train[batch:batch + batch_size])
				losses.append(loss)

			loss = np.mean(losses)

			[rmse,error] = test_model(self.users_test,self.items_test,self.ratings_test)
			mae = np.mean(np.abs(error))


			self.rmse_ = min(self.rmse_,rmse)
			if True:
				print('epoch : %d , loss : %f , rmse : %f, mae : %f' % (1 + epoch,loss,rmse,mae))
			

	def RMSE(self):
		return self.rmse_


class ml_100k(object):
	def __init__(self,directory,fold_count = 5):
		super(ml_100k, self).__init__()

		ratings =  pd.read_csv(directory,sep = '\t',header = None,engine='python')
		ratings.columns = ['userid','movieid','rating','timestamp']
		numUsers = 1 + ratings.userid.max()
		numItems = 1 + ratings.movieid.max()
		self.rating_matrix = np.zeros(shape = (numUsers,numItems),dtype = np.int32)
		self.rating_binary = np.zeros(shape = (numUsers,numItems),dtype = np.int32)
		


		#data split
		temp = []
		m = {}
		for it in ratings.itertuples():
			uid = getattr(it,'userid')
			iid = getattr(it,'movieid')
			rat = getattr(it,'rating')
			temp.append((uid,iid,rat))
			self.rating_matrix[uid,iid] = rat
			self.rating_binary[uid,iid] = 1

		self.test_binary = [np.zeros(shape = (numUsers,numItems),dtype = np.int32),
		np.zeros(shape = (numUsers,numItems),dtype = np.int32),
		np.zeros(shape = (numUsers,numItems),dtype = np.int32),
		np.zeros(shape = (numUsers,numItems),dtype = np.int32),
		np.zeros(shape = (numUsers,numItems),dtype = np.int32)]
		shuffle(temp)
		sep = len(temp) // fold_count

		
		for i in range(fold_count):
			for uid,iid,rat in temp[i*sep:(i+1)*sep]:
				self.test_binary[i][uid,iid] = 1

		self.numUsers, self.numItems = self.rating_matrix.shape
	def get_rating_matrix(self):	
		return self.rating_matrix

	def shape(self):
		return self.rating_matrix.shape

	def get_ith_train_test(self,fold):
		train_matrix = np.zeros(shape = self.rating_binary.shape, dtype = np.int32)
		test_matrix = self.test_binary[fold]
		for u in range(train_matrix.shape[0]):
			for j in range(train_matrix.shape[1]):
				train_matrix[u,j] = self.rating_binary[u,j]
				if test_matrix[u,j] == 1:
					train_matrix[u,j] = 0
		u_train = []
		u_test = []
		i_train = []
		i_test = []
		r_train = []
		r_test = []
		train = []
		test = []
		for u in range(self.rating_matrix.shape[0]):
			for i in range(self.rating_matrix.shape[1]):
				rat = self.rating_matrix[u,i]			
				if train_matrix[u,i] > 0:
					train.append((u,i,rat))
				elif test_matrix[u,i] > 0:
					test.append((u,i,rat))
		for u in range(self.rating_matrix.shape[0]):
			for i in range(self.rating_matrix.shape[1]):
				rat = self.rating_matrix[u,i]			
				if train_matrix[u,i] > 0:
					train.append((u,i,rat))
				elif test_matrix[u,i] > 0:
					test.append((u,i,rat))
		shuffle(train)
		shuffle(test)
		return train,test


if __name__ == "__main__":

	reg = 0.02
	latent_dim = 10
	learning_rate = 0.001
	iter = 20
	batch_size = 512
	rmse = []

	data = ml_100k('data/ml-100k/u.data')
	num_users,num_items = data.shape()
	for i in [0,1,2,3,4]:
		import pickle
		
		train,test =  data.get_ith_train_test(i)
		u_train,i_train,r_train = zip(*train)
		u_test,i_test,r_test = zip(*test)



		svd = SVD(u_train,u_test,i_train,i_test,r_train,r_test,latent_dim,learning_rate,reg)
		svd.train(iter,batch_size)
		print(svd.RMSE())
		rmse.append(svd.RMSE())
		print("predicted ratings")
		r_pred = svd.predict(u_test[25:30],i_test[25:30])
		r = r_test[25:30]
		print(r)
		print(r_pred)
	
	print("RMSE on folds")
	print(rmse)
	print("mean")
	print(np.mean(rmse))