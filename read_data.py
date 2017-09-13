import numpy as np 
import pandas as pd 
import os
from scipy import sparse

def dataset(DATA_DIR = 'data/ml-100k/preprocessed/',fold = 1,binarize = True):
	uid_fname='unique_uid_%d'%fold
	sid_fname= 'unique_sid_%d'%fold
	train_fname= 'train_%d.csv'%fold
	vad_fname= 'validation_%d.csv'%fold
	test_fname = 'test_%d.csv'%fold

	unique_uid = list()
	with open(os.path.join(DATA_DIR,uid_fname), 'r') as f:
	    for line in f:
	        unique_uid.append(line.strip())
	
	unique_sid = list()
	with open(os.path.join(DATA_DIR, sid_fname), 'r') as f:
	    for line in f:
	        unique_sid.append(line.strip())

	unique_uid = [int(x) for x in unique_uid]
	unique_sid = [int(x) for x in unique_sid]

	n_items = len(unique_sid)
	n_users = len(unique_uid)
	print(n_items,n_users)

	def load_data(csv_file, shape=(n_users, n_items),binarize = True):
	    tp = pd.read_csv(csv_file)
	    rows, cols, ratings = np.array(tp['uid']), np.array(tp['sid']), np.array(tp['rating'])
	    
	    seq = np.concatenate((rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'),ratings[:,None]), axis=1)
	    if binarize:
	    	data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), shape=shape)
	    else:
	    	data = sparse.csr_matrix((ratings, (rows, cols)), shape=shape)
	    return data, seq


	train_data, train_raw = load_data(os.path.join(DATA_DIR, train_fname),binarize = binarize)
	vad_data, vad_raw = load_data(os.path.join(DATA_DIR,vad_fname),binarize = binarize)
	test_data,test_raw = load_data(os.path.join(DATA_DIR,test_fname),binarize = binarize)
	return train_data,train_raw,vad_data,vad_raw,test_data,test_raw,unique_uid,unique_sid


