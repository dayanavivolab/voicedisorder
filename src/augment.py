import numpy as np 
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from collections import Counter

'''
Description: Functions for data augmentation to deal with class imbalance 
Smote algorithm
https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

Copyright: Vivolab, 2022
contact: dribas@unizar.es
'''

def augment_smote_balance(train_features, train_labels, debug=False):
    
    counter = Counter(train_labels)
    if debug:
        plt.figure(1)
        plt.subplot(121)
        for label, _ in counter.items():
            row_ix = np.where(train_labels == label)[0]
            # scatter de los dos primeros parametros: param0 vs. param1
            plt.scatter(train_features[row_ix, 0], train_features[row_ix, 1], label=str(label))
        plt.xlabel('feature1')
        plt.ylabel('feature2') 
        plt.legend()
    
    # Smote1: Balance classes
    oversample = SMOTE()
    train_features, train_labels = oversample.fit_resample(train_features, train_labels)
    
    if debug: 
        plt.subplot(122)
        for label, _ in counter.items():
            row_ix = np.where(train_labels == label)[0]
            plt.scatter(train_features[row_ix, 0], train_features[row_ix, 1], label=str(label))
        plt.xlabel('feature1')
        plt.ylabel('feature2') 
        plt.legend()
        plt.show()
    
    return train_features, train_labels

def augment_borderlinesmote_balance(train_features, train_labels):
    oversample = BorderlineSMOTE()
    train_features, train_labels = oversample.fit_resample(train_features, train_labels)
    return train_features, train_labels

def augment_svmsmote_balance(train_features, train_labels):
    oversample = SVMSMOTE()
    train_features, train_labels = oversample.fit_resample(train_features, train_labels)
    return train_features, train_labels

def augment_adasynsmote_balance(train_features, train_labels):
    oversample = ADASYN()
    train_features, train_labels = oversample.fit_resample(train_features, train_labels)
    return train_features, train_labels

def augment_smote(train_features, train_labels, val=10, debug=False):
    
    counter = Counter(train_labels)
    print('1.Init: Norm: %i, Path: %i' % (counter[0], counter[1]))
    if counter[0]>counter[1]:
        mayor = counter[0]
    else:
        mayor = counter[1]
    total = mayor*val

    while mayor<total:
        train_features, train_labels = augment_smote_balance(train_features, train_labels, debug=False)
        counter = Counter(train_labels)
        print('2a.Balancing: Norm: %i, Path: %i' % (counter[0], counter[1]))
        siz = counter[0]
        
        ind = np.where(train_labels==1)[0]
        X1 = train_features[ind]
        y1 = train_labels[ind]
        s = int(np.floor(siz/val))
        ind = np.where(train_labels==0)[0][0:s]
        X0 = train_features[ind]
        y0 = train_labels[ind]

        X = np.concatenate((X0, X1))
        y = np.concatenate((y0,y1),axis=0)
        
        X, y = augment_smote_balance(X, y, debug=False)
        ind = np.where(y==0)[0][s:]
        X0 = X[ind]
        y0 = y[ind]

        train_features = np.concatenate((train_features, X0))
        train_labels = np.concatenate((train_labels, y0))
        counter = Counter(train_labels)
        print('2b.Augmenting: Norm: %i, Path: %i' % (counter[0], counter[1]))
        mayor = counter[0]

    train_features, train_labels = augment_smote_balance(train_features, train_labels, debug=False)
    counter = Counter(train_labels)
    print('3.Output: Norm: %i, Path: %i' % (counter[0], counter[1]))

    return train_features, train_labels
