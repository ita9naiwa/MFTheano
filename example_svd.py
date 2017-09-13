
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

import theano
import sklearn.utils
import algorithms.svd as svd
import evaluation.rec_eval as rec_eval
from read_data import *

train_data,train_raw,vad_data,vad_raw,test_data,test_raw,unique_uid,unique_iid = dataset(binarize = False)
n_items = len(unique_iid)
n_users = len(unique_uid)

train_raw = train_raw.astype(np.int32)
vad_raw = vad_raw.astype(np.int32)
u_train = train_raw[:,0]
i_train = train_raw[:,1]
r_train = train_raw[:,3]
u_vad = vad_raw[:,0]
i_vad = vad_raw[:,1]
r_vad = vad_raw[:,3]

latent_dim = 20
learning_rate = 0.002
epochs = 50
reg = 0.01

biased_MF = svd.SVD(n_users,n_items,latent_dim,learning_rate,reg)
    #__init__(self,unique_users,unique_items,latent_dim = 16,learning_rate = 0.01,reg = 0.01):
for i in range(epochs):
    u_train,i_train,r_train = sklearn.utils.shuffle(u_train,i_train,r_train)
    loss_mean = biased_MF.train(u_train,i_train,r_train)
    test_rmse = biased_MF.RMSE(u_vad,i_vad,r_vad)
    print(loss_mean,test_rmse)

    
    
    
    
    