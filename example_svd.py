
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
from read_data import *

train_data,train_raw,vad_data,vad_raw,test_data,test_raw,unique_uid,unique_iid = dataset()
n_items = len(unique_iid)
n_users = len(unique_uid)

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
    