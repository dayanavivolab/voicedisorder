'''
Description: Executable for run the Voice Disorder Detection system (binary classification)
             for SVD (http://www.stimmdatenbank.coli.uni-saarland.de/help_en.php4) 
             and AVFAD (http://acsa.web.ua.pt/AVFAD.htm) databases.
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
from collections import Counter


def main(list_path, kfold, audio_type):

    # 0. Config and output logs
    ker, d, c ='poly',1,1 # SVM configuration
    label = os.path.basename(list_path)

    respath = 'data/result/'+label
    if not os.path.exists(respath): os.mkdir(respath)

    result_log = respath+'/results_'+label+'_'+audio_type+'_'+ker+str(d)+'c'+str(c)+'.log'
    f = open(result_log, 'w+')
    f.write('Results Data:%s Features:Compare2016 %ifold, %s\n' % (label, kfold, audio_type))
    f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f \n' % (ker, d, c))
    f.close()

       
    score = np.zeros((13,kfold))
    score_oracle = np.zeros((13,kfold))

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
        # Get the same features.pkl for binaryclass and for multiclass 
        try: audio_type_pkl = audio_type.split('multi_')[1] 
        except: audio_type_pkl = audio_type
        try: label_csv = label.split('_Nomiss')[1] 
        except: label_csv = label

        train_labels = np.array(train_labels)
        if not os.path.exists('data/features/'+label): os.mkdir('data/features/'+label)
        trainpath = 'data/features/'+label+'/train_'+audio_type_pkl+'_fold'+str(k+1)+'.pkl'
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
                feat = pd.read_csv('data/features/'+label_csv+'/'+name+'_smile.csv').to_numpy()[0]
                train_features.append(feat[3:])
                i=i+1
            print('Train: ' + str(i))          
            train_features = np.array(train_features)
            with open(trainpath,'wb') as fid:
                pickle.dump(train_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
        train_features, trainmean, trainstd = zscore(train_features)
        # Test
        test_labels = np.array(test_labels)
        testpath = 'data/features/'+label+'/test_'+audio_type_pkl+'_fold'+str(k+1)+'.pkl'
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
                feat = pd.read_csv('data/features/'+label_csv+'/'+name+'_smile.csv').to_numpy()[0]
                test_features.append(feat[3:])
                i=i+1
            print('Test: ' + str(i))
            test_features = np.array(test_features) 
            with open(testpath,'wb') as fid:
                pickle.dump(test_features, fid, protocol=pickle.HIGHEST_PROTOCOL)            
        test_features = zscore(test_features, trainmean, trainstd)


        # 3. Train SVM classifier
        counter = Counter(train_labels)
        print('HEALTH: %i, PATH: %i\n' % (counter[0], counter[1]))
        
        clf = SVC(C=c,kernel=ker,degree=d,probability=True)
        clf.fit(train_features, train_labels)

        # 4. Testing
        out = clf.predict(test_features)
        out_oracle = clf.predict(train_features)

        score[:,k] = compute_score(clf, test_labels, out, test_features)
        with open (respath+'/output_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
            lbl = ['HEALTH','PATH']
            for j in range(0,len(test_files)):
                f.write('%s %s %s\n' % (os.path.basename(test_files[j])[:-4], lbl[test_labels[j]], lbl[out[j]]))
        
        score_oracle[:,k] = compute_score(clf, train_labels, out_oracle, train_features)
        with open (respath+'/output_oracle_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
            lbl = ['HEALTH','PATH']
            for j in range(0,len(train_files)):
                f.write('%s %s %s\n' % (os.path.basename(train_files[j])[:-4], lbl[train_labels[j]], lbl[out_oracle[j]]))
        

        toc = time.time()
        f = open(result_log, 'a')
        f.write('Oracle Fold%i (%.2fsec): Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
        (k+1, toc-tic, score_oracle[0,k], score_oracle[1,k], score_oracle[2,k], score_oracle[3,k], score_oracle[4,k], score_oracle[5,k], score_oracle[6,k], score_oracle[7,k], score_oracle[8,k], 
        score_oracle[9,k], score_oracle[10,k], score_oracle[11,k], score_oracle[12,k]))
        f.close()
        toc = time.time()
        f = open(result_log, 'a')
        f.write('Test Fold%i (%.2fsec): Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n\n' % 
        (k+1, toc-tic, score[0,k], score[1,k], score[2,k], score[3,k], score[4,k], score[5,k], score[6,k], score[7,k], score[8,k], score[9,k], score[10,k], score[11,k], score[12,k]))
        f.close()

    # Mean
    f = open(result_log, 'a')
    f.write('\nMean among folds')
    f.write('TOTAL Oracle: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.mean(score_oracle[0,:]), np.mean(score_oracle[1,:]), np.mean(score_oracle[2,:]), np.mean(score_oracle[3,:]), np.mean(score_oracle[4,:]), np.mean(score_oracle[5,:]), 
        np.mean(score_oracle[6,:]), np.mean(score_oracle[7,:]), np.mean(score_oracle[8,:]), np.mean(score_oracle[9,:]), np.mean(score_oracle[10,:]), np.mean(score_oracle[11,:]), np.mean(score_oracle[12,:])))
    f.close()

    f = open(result_log, 'a')
    f.write('TOTAL Test: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n\n' % 
    (np.mean(score[0,:]), np.mean(score[1,:]), np.mean(score[2,:]), np.mean(score[3,:]), np.mean(score[4,:]), np.mean(score[5,:]), 
        np.mean(score[6,:]), np.mean(score[7,:]), np.mean(score[8,:]), np.mean(score[9,:]), np.mean(score[10,:]), np.mean(score[11,:]), np.mean(score[12,:])))
    f.close()

    # Variance
    f = open(result_log, 'a')
    f.write('\nStandard deviation among folds')
    f.write('TOTAL Oracle: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.std(score_oracle[0,:]), np.std(score_oracle[1,:]), np.std(score_oracle[2,:]), np.std(score_oracle[3,:]), np.std(score_oracle[4,:]), np.std(score_oracle[5,:]), 
        np.std(score_oracle[6,:]), np.std(score_oracle[7,:]), np.std(score_oracle[8,:]), np.std(score_oracle[9,:]), np.std(score_oracle[10,:]), np.std(score_oracle[11,:]), np.std(score_oracle[12,:])))
    f.close()

    f = open(result_log, 'a')
    f.write('TOTAL Test: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n\n' % 
    (np.std(score[0,:]), np.std(score[1,:]), np.std(score[2,:]), np.std(score[3,:]), np.std(score[4,:]), np.std(score[5,:]), 
        np.std(score[6,:]), np.std(score[7,:]), np.std(score[8,:]), np.std(score[9,:]), np.std(score[10,:]), np.std(score[11,:]), np.std(score[12,:])))
    f.close()



    




#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print('audio_type (SVD): a_n, aiu, phrases, multi_a_n, multi_aiu, multi_phrases')
        print('audio_type (AVFAD): aiu, phrases, read, spontaneous, multi_aiu, multi_phrases, multi_read, multi_spontaneous')
        print('Usage: run_voicedisorder.py list_path kfold audio_type')
        
        print('Example: python run_voicedisorder.py data/lst 5 phrase_both')
    else:
        list_path = args[0]
        kfold = int(args[1])
        audio_type = args[2]
        main(list_path, kfold, audio_type)
