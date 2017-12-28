"""
@author : Hyunsung lee
@email : ita9naiwa@gmail.com
"""

from .Recommender import ImplicitRecommender

import json
import logging
from  gensim.models import Word2Vec

import os,sys
import time

class Shelf2Vec(object):
	def __init__(self,model_config = None,
		db_config = None,log_dir = 'log'):
		logging.basicConfig(filename = os.path.join(log_dir,'log'),level = logging.INFO)
		logger = logging.getLogger('shelf2vec')
		self.model = None
		self.model_config = model_config
		self.db_config = db_config
		print(self.model_config)
		print(self.db_config)
		self.model_begin()

	def model_begin(self):
		model_config 	= self.model_config
		db_config		= self.db_config
		dataset = self.load_dataset(db_config)
		logger.info("training begins...")
		if model_config is not None:
			self.model = Word2Vec(dataset,
			size = model_config['latent_dim'],
			min_count = model_config['min_count'],
			seed = model_config['seed'],
			alpha = model_config['learning_rate'],
			workers = model_config['n_jobs'],
			negative = model_config['n_negatives'],
			iter = model_config['n_iter'],
			window = 123456789,
			sg = 1)
		else:
			# Debug statement:
			self.model = Word2Vec(dataset,
				size = 2,
				min_count = 100,
				seed = 0,
				iter = 1)
			return False
		logger.info("training done...")
		return True

	def most_similar(self,shelf,k=2):
		if self.model == None:
			return False
		else:
			return list(zip(*self.model.wv.most_similar(shelf,topn=k)))

	def load_dataset(self,db_config):
		logger.info("data load...")
		if db_config == None:
			logging.error("no db information given")
			return None
		logger.info("sending query")
		query = """select tbp.book_name,tbp.book_id,tbp.bookshelf_id,tbc.category_id
    		From (select book_name, book_id, bookshelf_id from tbl_bookshelf_piece where bookshelf_id is not null) tbp
    		Inner Join (select book_id, category_id from tbl_book_category) tbc
    		On tbp.book_id = tbc.book_id"""
		conn = pymysql.connect(host=db_config['db_host'],port=db_config['db_port'], user=db_config['db_user'], password=db_config['db_password'],
		                       db=db_config['db_database'], charset='utf8')
		curs = conn.cursor()
		# SQL문 실행
		raw_data = pd.read_sql(query , con=conn)
		logger.info("get data from db")
		#This is quite slow, is there any faster way to do this?
		#make list of books in the same shelf
		#item_id,shelf_id rows -> shelf_1 : [item_1,item_2,...,]
		return raw_data.groupby('bookshelf_id').book_name.apply(list).value



class Prod2vec(ImplicitRecommender):
	def __init__(self,dtype = 'float32',verbose = True,seed=None,**kwargs):
		super(Prod2vec,self).__init__(dtype,verbose,seed,**kwargs)
		self.model_name = 'Prod2vec'
		self.seed = seed
		self._parse_kwargs(**kwargs)
	def _parse_kwargs:
		self.latent_dim = int32(kwargs.get('latent_dim',100))
		# items which appears less than min_count should be erased first
		self.min_count = 0
		self.