    import bottleneck as bn
import numpy as np
import bottleneck as bn
#hmm
from scipy import sparse

def precision_recall_at_ks(model,train_data,test_data,ks = [1,3,5,10,20,50]):
    X_pred = model.predict(train_data)
    np.max(ks)
    precs = []
    recalls = []
    for k in ks:
        idx = np.argpartition(X_pred,k,axis = 1)
        X_pred_binary = np.zeros_like(test_data,dtype = bool)
        X_pred_binary[np.arange(train_data.shape[0])[:, np.newaxis], idx[:, :k]] = True
        tmp = (np.logical_and(test_data, X_pred_binary).sum(axis=1)).astype(
            np.float32)
        precs.append(np.nanmean(tmp/k))
        recalls.append(np.nanmean(tmp/test_data.sum(axis=1)))
    return precs,recalls
