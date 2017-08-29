
# coding: utf-8

import itertools
import glob
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import numpy as np

import pandas as pd
from scipy import sparse
import importlib

from random import shuffle
import theano
import sklearn.utils
import algorithms.bpr as bpr
import evaluation.rec_eval as rec_eval



DATA_DIR = 'data/ml-100k/processed/'
fold = 1
uid_fname='unique_uid_%d'%fold
sid_fname= 'unique_sid_%d'%fold
train_fname= 'train_%d.csv'%fold
vad_fname= 'validation_%d.csv'%fold
test_fname = 'test_%d.csv'%fold


unique_uid = list()
with open(os.path.join('data/ml-100k/processed/',uid_fname), 'r') as f:
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

def load_data(csv_file, shape=(n_users, n_items)):
    tp = pd.read_csv(csv_file)
    timestamps, rows, cols = np.array(tp['timestamp']), np.array(tp['uid']), np.array(tp['sid'])
    seq = np.concatenate((rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'), timestamps[:, None]), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq


train_data, train_raw = load_data(os.path.join(DATA_DIR, train_fname))
watches_per_movie = np.asarray(train_data.astype('int64').sum(axis=0)).ravel()
print("The mean (median) watches per movie is %d (%d)" % (watches_per_movie.mean(), np.median(watches_per_movie)))
user_activity = np.asarray(train_data.sum(axis=1)).ravel()
print("The mean (median) movies each user wathced is %d (%d)" % (user_activity.mean(), np.median(user_activity)))


vad_data, vad_raw = load_data(os.path.join(DATA_DIR,vad_fname))
test_data,test_raw = load_data(os.path.join(DATA_DIR,test_fname))


u_train = train_raw[:,0]
i_train = train_raw[:,1]
u_vad = vad_raw[:,0]
i_vad = vad_raw[:,1]
train_map = set()
for i,j in zip(*[u_train,i_train]):
    train_map.add((i,j))
    
import sklearn.utils

def get_train_data_mf(cnt = -1):
    if cnt == -1:
        cnt = len(u_train) 
    sz = 100000
    users = []
    items = []
    neg_items = []
    
    
    def init(idx = 0,neg_samples = None):
        neg_samples = np.random.choice(np.arange(n_items),size = sz)
        idx = 0
        return idx,neg_samples
    idx,neg_samples = init()
    l = len(u_train)
    t = []
    indices = np.random.choice(np.arange(l),size = cnt)
    for i in indices:
        user = u_train[i]
        item = i_train[i]
        if idx == sz:
            idx,neg_samples = init()
        neg_item = neg_samples[idx]
        idx+=1
        while (user,neg_item) in train_map:
            if idx == sz:
                idx,neg_samples = init()
            neg_item = neg_samples[idx]
            idx+=1
        users.append(user)
        items.append(item)
        neg_items.append(neg_item)

    users,items,neg_items = sklearn.utils.shuffle(users,items,neg_items)
    return users,items,neg_items


latent_dim = 10
learning_rate = 0.05
val_decrease = 0
val_dec_max = 5
epochs = 150
best_ndcg = -1
test_prec = -1
bpr_mf = bpr.BPR_MF(n_users,n_items,latent_dim,learning_rate,1e-4*5.0)
best_idx = -1
cnt = 0
last_val = 0.00
for i in range(epochs):
    a,b,c = get_train_data_mf(100000)
    
    loss = bpr_mf.train_mf(a,b,c)
    U,V = bpr_mf.get_params()
    val_ndcg = rec_eval.normalized_dcg_at_k(train_data, vad_data, U, V, k=50, vad_data=None)
    if best_ndcg < val_ndcg:
       best_ndcg = val_ndcg
    test_prec = rec_eval.prec_at_k(train_data, test_data, U, V, k=10, vad_data=vad_data)
    print('best test test_prec : %f'% test_prec)
    
    if val_ndcg > last_val:
        val_decrease = 0
        last_val = val_ndcg
        bpr_mf.reset_lr(q = 1.2)
    else:
        last_val = val_ndcg
        if bpr_mf.lr*0.5 >= 0.01:
            bpr_mf.reset_lr(q = 0.5)
        elif bpr_mf.lr > 0.01:
            bpr_mf.reset_lr(p = 0.01)
        val_decrease+=1
        if val_decrease == val_dec_max:
            break

print('best test precision@10' % test_prec)
    