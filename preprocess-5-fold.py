# coding: utf-8
# mainly brought  Liang's work
# https://github.com/dawenl/cofactor

import datetime
import json
import os
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import scipy.sparse

import seaborn as sns
sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')

DATA_DIR = 'data/ml-100k/'
fc = 5
np.random.seed(1541)




def timestamp_to_date(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

#ml-20m
#raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
#ml-1m
#raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'),sep = '::',engine='python')
#ml-100k
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'u.data'),sep = '\t',header=None)



raw_data.columns = ['uid','sid','rating','timestamp']
raw_data.reset_index(inplace=True,drop=True)


tr_vd_raw_data = raw_data


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=0, min_sc=0):
    # Only keep the triplets for songs which were listened to by at least min_sc users. 
    if min_sc > 0:
        songcount = get_count(tp, 'sid')
        tp = tp[tp['sid'].isin(songcount.index[songcount >= min_sc])]
    
    # Only keep the triplets for users who listened to at least min_uc songs
    # After doing this, some of the songs will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'sid')
        tp = tp[tp['uid'].isin(usercount.index[usercount >= min_uc])]
    
    # Update both usercount and songcount after filtering
    usercount, songcount = get_count(tp, 'uid'), get_count(tp, 'sid') 
    return tp, usercount, songcount

tr_vd_raw_data, user_activity, item_popularity = filter_triplets(tr_vd_raw_data)

sparsity = 1. * tr_vd_raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
      (tr_vd_raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))


unique_uid = user_activity.index
unique_sid = item_popularity.index


n_ratings = tr_vd_raw_data.shape[0]
test = np.random.choice(n_ratings, size=(n_ratings), replace=False)
tests = []

split = n_ratings // fc
for i in range(fc):
    tests.append(test[split*i:split*(i+1)])



tr_vad_raw_datas = []
train_raw_datas = []
vad_raw_datas = []
test_raw_datas = []

for i in range(fc):
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[tests[i]] = True
    test_raw_datas.append(tr_vd_raw_data[test_idx])
    tr_vad_raw_datas.append(tr_vd_raw_data[~test_idx])
    tr_vad_n_ratings = tr_vad_raw_datas[i].shape[0]
    vad_idx = np.zeros(tr_vad_n_ratings, dtype=bool)
    vad = np.random.choice(tr_vad_n_ratings, size=(tr_vad_n_ratings//10), replace=False)
    vad_idx[vad] = True
    train_raw_datas.append(tr_vad_raw_datas[i][~vad_idx])
    vad_raw_datas.append(tr_vad_raw_datas[i][vad_idx])
        
    

    
print(len(test_raw_datas))


def numerize(tp):
    uid = list(map(lambda x: user2id[x], tp['uid']))
    sid = list(map(lambda x: song2id[x], tp['sid']))
    
    tp['uid'] = uid
    tp['sid'] = sid
    return tp[[ 'uid', 'sid','rating','timestamp']]


# In[171]:

directory = 'processed'
for idx, items in enumerate(zip(*[train_raw_datas,vad_raw_datas,test_raw_datas])):
    train_raw_data,vad_raw_data,test_raw_data = items

    train_sid = set(pd.unique(train_raw_data['sid']))
    train_uid = set(pd.unique(train_raw_data['uid']))
    left_sid = list()
    for i, sid in enumerate(unique_sid):
        if sid not in train_sid:
            left_sid.append(sid)
    left_uid = list()
    for i, uid in enumerate(unique_uid):
        if uid not in train_uid:
            left_uid.append(uid)
    move_idx = vad_raw_data['sid'].isin(left_sid)
    user_idx = vad_raw_data['uid'].isin(left_uid)
    train_raw_data = train_raw_data.append(vad_raw_data[move_idx])
    train_raw_data = train_raw_data.append(vad_raw_data[user_idx])
    vad_raw_data = vad_raw_data[~move_idx]
    vad_raw_data = vad_raw_data[~user_idx]
    test_raw_data = test_raw_data[test_raw_data['sid'].isin(train_sid)]
    test_raw_data = test_raw_data[test_raw_data['uid'].isin(train_uid)]
    
    
    print ("There are total of %d unique users in the training set and %d unique users in the fold %d dataset" % \
    (len(pd.unique(train_raw_data['uid'])), len(unique_uid),idx))
    print ("There are total of %d unique items in the vad set and %d unique items in the fold %d dataset" % \
    (len(pd.unique(vad_raw_data['sid'])), len(unique_sid),idx))
    print ("There are total of %d unique items in the test set and %d unique items in the fold %d dataset" % \
    (len(pd.unique(test_raw_data['sid'])), len(unique_sid),idx))
    print("train data length  %d , test data length %d, validation data length %d\n\n" % (len(train_raw_data),len(test_raw_data),len(vad_raw_data)))
    song2id = dict((sid, i) for (i, sid) in enumerate(train_raw_data['sid'].unique()))
    user2id = dict((uid, i) for (i, uid) in enumerate(train_raw_data['uid'].unique()))
    r = 0

    with open(os.path.join(DATA_DIR, directory, 'unique_uid_%d' % idx), 'w') as f:
        for i,uid in enumerate(unique_uid):
            r = max(i,r)
            f.write('%s\n' % i)
    r = 0
    with open(os.path.join(DATA_DIR, directory, 'unique_sid_%d' % idx), 'w') as f:
        for i,sid in enumerate(unique_sid):
            r = max(i,r)
            f.write('%s\n' % i)
    assert r+1 == len(unique_sid)
    train_data = numerize(train_raw_data)
    train_data.to_csv(os.path.join(DATA_DIR, directory, 'train_%d.csv' % idx), index=False)

    vad_data = numerize(vad_raw_data)
    vad_data.to_csv(os.path.join(DATA_DIR, directory, 'validation_%d.csv' % idx ), index=False)
    
    test_data = numerize(test_raw_data)
    test_data.to_csv(os.path.join(DATA_DIR, directory, 'test_%d.csv' % idx ), index=False)
    
