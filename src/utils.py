import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import *
import matplotlib.pyplot as plt

'''
Description: Utils functions for voice disorder detection 

Copyright: Vivolab, 2022
contact: dribas@unizar.es
'''

def zscore(x,u=False,s=False):
    sx,feat = x.shape
    xnorm = np.zeros((sx,feat))
    try:
        test = u.shape #Just for invalidate the try if u==False
        xnorm = (x - u)/s
        return xnorm
    except: 
        u = np.mean(x,axis=0)
        s = np.zeros(feat)
        for i in range(0,feat):
            s[i] = np.std(x[:,i]) + 1e-20    
        xnorm = (x - u)/s
        return xnorm, u, s
        
def compute_score(model, test_lbltrue, test_lblpredict, test_features, roc=False):
    # output: score (has 13 metrics values)  
    # score = 0.acc, 1.acc0, 2.acc1, 3.uar, 4.f1score, 5.recall, 6.precision, 7.auc, 8.eer, 9.tp, 10.tn, 11.fp, 12.fp
    score = []
    score.append(accuracy_score(test_lbltrue, test_lblpredict))
    score.append(accuracy_score(test_lbltrue[test_lbltrue==0], test_lblpredict[test_lbltrue==0]))
    score.append(accuracy_score(test_lbltrue[test_lbltrue==1], test_lblpredict[test_lbltrue==1]))
    score.append(balanced_accuracy_score(test_lbltrue, test_lblpredict))
    score.append(f1_score(test_lbltrue,test_lblpredict))
    score.append(recall_score(test_lbltrue,test_lblpredict))
    score.append(precision_score(test_lbltrue,test_lblpredict))
    scores=model.predict_proba(test_features)[:,1]
    fpr, tpr, thresholds = roc_curve(test_lbltrue, scores)
    score.append(roc_auc_score(test_lbltrue, scores))
    eer_value=brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    score.append(eer_value)
    if roc:
        plt.figure()
        plt.plot(fpr,tpr)
        plt.title('EER='+str(eer_value))
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.grid(True)
        plt.savefig('roc_curve.png')
    tn, fp, fn, tp = confusion_matrix(test_lbltrue, test_lblpredict).ravel()
    N = tn + fp + fn + tp
    score.append(tp/N * 100)
    score.append(tn/N * 100)
    score.append(fp/N * 100)
    score.append(fn/N * 100) 
    return np.array(score) 
    
    
def compute_score_multiclass(test_lbltrue, test_lblpredict, data):
    
    labelsnames = data['labels'].keys()
    lbl = {}
    for i in labelsnames:
        lbl[data['labels'][i]] = i 

    labels = np.unique(test_lbltrue)
    target_names = []
    for i in labels:
        target_names.append(lbl[i])
    out = classification_report(test_lbltrue, test_lblpredict, labels=labels, target_names=target_names)

    return out

