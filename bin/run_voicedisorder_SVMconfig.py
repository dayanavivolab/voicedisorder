'''
Description: Executable for run the Voice Disorder Detection system (binary classification)
             for SVD (http://www.stimmdatenbank.coli.uni-saarland.de/help_en.php4).

             This is a swept of parameters for SVM classifier configuration.

System: 
- Frontend: Opensmile (Compare2016) features  https://audeering.github.io/opensmile-python/
- Backend: SVM classifier with polinomial kernel, d=1, c=1  https://scikit-learn.org/stable/
- Performance metrics: https://scikit-learn.org/stable/

Copyright: Vivolab, 2022
contact: dribas@unizar.es
'''

import sys, json, os, pickle, time
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.svm import SVC

from utils import compute_score, zscore


def main(list_path, kfold, audio_type):
    
    # acc, uar, f1score, recall, precision, auc
    f = open('results_'+audio_type+'.log', 'w+')
    f.write('Results Compare2016 %ifold, %s\n' % (kfold, audio_type))
    f.close()

    Clist = [1, 0.1, 0.5, 1.5, 2, 3, 4, 5, 6]
    Dlist = [1, 2, 3]
    Klist = ['poly', 'rbf']
    Glist = [1/6373]
    if not os.path.exists('data/result'): os.mkdir('data/result')
    
    # Swept of SVM classifier parameters
    for ker in Klist: 
        for d in Dlist: 
            for c in Clist:
                for g in Glist:
                    f = open('results_'+audio_type+'.log', 'a')
                    f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f, Gamma=%.2f \n' % (ker, d, c, g))
                    f.close()
                    score = np.zeros((12,kfold))
                    # 1. Loading data from json list
                    for k in range(0,kfold):
                        tic = time.time()
                        train_files = [] 
                        train_labels = [] 
                        trainlist = list_path + '/train_' + audio_type + '_meta_data_fold' + str(k+1) + '.json'
                        with open(trainlist, 'r') as f:
                            data = json.load(f)
                            for item in data['meta_data']:
                                train_files.append(item['path'])
                                if item['label']=='HEALTH':
                                    train_labels.append(0)
                                else:
                                    train_labels.append(1)
                        f.close()

                        test_files = [] 
                        test_labels = [] 
                        testlist = list_path + '/test_' + audio_type + '_meta_data_fold' + str(k+1) + '.json'
                        with open(testlist, 'r') as f:
                            data = json.load(f)
                            for item in data['meta_data']:
                                test_files.append(item['path'])
                                if item['label']=='HEALTH':
                                    test_labels.append(0)
                                else:
                                    test_labels.append(1)
                        f.close()

    
                        # 2. Load features: Train
                        train_labels = np.array(train_labels)
                        trainpath = 'data/features/train_'+audio_type+'_fold'+str(k+1)+'.pkl'
                        if os.path.exists(trainpath): 
                            with open(trainpath,'rb') as fid:
                                train_features = pickle.load(fid)
                                print('Fold '+ str(k+1) +' Train: ' + str(train_features.shape))
                        else:
                            i=0
                            train_features = []
                            for wav in train_files:
                                print(str(i) + ': Fold ' + str(k+1) + ': ' + wav)
                                name = os.path.basename(wav)[:-4]
                                feat = pd.read_csv('data/features/'+name+'_smile.csv').to_numpy()[0]
                                train_features.append(feat[3:])
                                i=i+1
                            print('Train: ' + str(i))          
                            train_features = np.array(train_features)
                            with open(trainpath,'wb') as fid:
                                pickle.dump(train_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
                        train_features, trainmean, trainstd = zscore(train_features)
                        # Test
                        test_labels = np.array(test_labels)
                        testpath = 'data/features/test_'+audio_type+'_fold'+str(k+1)+'.pkl'
                        if os.path.exists(testpath): 
                            with open(testpath,'rb') as fid:
                                test_features = pickle.load(fid)
                                print('Fold '+ str(k+1) + ' Test: ' + str(test_features.shape))
                        else:
                            i=0
                            test_features = []
                            for wav in test_files:
                                print(str(i) + ': Fold ' + str(k+1) + ': ' + wav)
                                name = os.path.basename(wav)[:-4]          
                                feat = pd.read_csv('data/features/'+name+'_smile.csv').to_numpy()[0]
                                test_features.append(feat[3:])
                                i=i+1
                            print('Test: ' + str(i))
                            test_features = np.array(test_features) 
                            with open(testpath,'wb') as fid:
                                pickle.dump(test_features, fid, protocol=pickle.HIGHEST_PROTOCOL)            
                        test_features = zscore(test_features, trainmean, trainstd)

        
                        # 3. Train SVM classifier
                        #clf = make_pipeline(SVC(C=1.0,kernel='poly',degree=1))
                        #clf.fit(train_features, train_labels)
                        #clf_norm = make_pipeline(StandardScaler(), SVC(C=1.0,kernel='poly',degree=1))
                        #clf_norm.fit(train_features, train_labels)
                        clf = SVC(C=c,kernel=ker,degree=d,gamma=g)
                        clf.fit(train_features, train_labels)

                        # 4. Testing
                        out = clf.predict(test_features)
        
                        score[:,k] = compute_score(clf, test_labels, out, test_features)
                        with open ('data/result/output_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'g'+str(g)+'.log', 'w') as f:
                            lbl = ['HEALTH','PATH']
                            for j in range(0,len(test_files)):
                                f.write('%s %s %s\n' % (os.path.basename(test_files[j])[:-4], lbl[test_labels[j]], lbl[out[j]]))
                        

                        toc = time.time()
                        f = open('results_'+audio_type+'.log', 'a')
                        f.write('Fold%i (%.2fsec): Acc=%0.2f, AccHealth=%0.2f, AccPath=%0.2f, UAR=%0.2f, F1Score=%0.2f, Recall=%0.2f, Precision=%0.2f, AUC=%0.2f, TP=%0.2f, TN=%0.2f, FP=%0.2f, FN=%0.2f, \n' % 
                        (k+1, toc-tic, score[0,k], score[1,k], score[2,k], score[3,k], score[4,k], score[5,k], score[6,k], score[7,k], score[8,k], score[9,k], score[10,k], score[11,k]))
                        f.close()

                    f = open('results_'+audio_type+'.log', 'a')
                    f.write('TOTAL: Acc=%0.2f, AccHealth=%0.2f, AccPath=%0.2f, UAR=%0.2f, F1Score=%0.2f, Recall=%0.2f, Precision=%0.2f, AUC=%0.2f, TP=%0.2f, TN=%0.2f, FP=%0.2f, FN=%0.2f, \n' % 
                    (np.mean(score[0,:]), np.mean(score[1,:]), np.mean(score[2,:]), np.mean(score[3,:]), np.mean(score[4,:]), np.mean(score[5,:]), 
                     np.mean(score[6,:]), np.mean(score[7,:]), np.mean(score[8,:]), np.mean(score[9,:]), np.mean(score[10,:]), np.mean(score[11,:])))
                    f.close()








    




#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print('Usage: run_voicedisorder_SVMconfig.py list_path kfold audio_type')
        print('Example: python run_voicedisorder_SVMconfig.py data/lst 5 aiu_n_both')
    else:
        list_path = args[0]
        kfold = int(args[1])
        audio_type = args[2]
        main(list_path, kfold, audio_type)
