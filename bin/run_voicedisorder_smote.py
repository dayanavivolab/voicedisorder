
'''
Description: Executable for run the Voice Disorder Detection system (binary classification)
             for SVD (http://www.stimmdatenbank.coli.uni-saarland.de/help_en.php4) 
             and AVFAD (http://acsa.web.ua.pt/AVFAD.htm) databases.
System: 
- Frontend: Opensmile (Compare2016) features  https://audeering.github.io/opensmile-python/
- Backend: SVM classifier with polinomial kernel, d=1, c=1  https://scikit-learn.org/stable/
- Data augmentation: Smote algorithm https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
- Performance metrics: https://scikit-learn.org/stable/

Copyright: Vivolab, 2022
contact: dribas@unizar.es
'''

import sys, gc, json, os, pickle, time
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import *
from collections import Counter

from utils import compute_score, zscore
import augment


def main(list_path, kfold, audio_type):
    
    # 0. Config and output logs
    ker, d, c ='poly',1,1 # SVM configuration
    label = os.path.basename(list_path)

    respath = 'data/result/'+label
    if not os.path.exists(respath): os.mkdir(respath)

    result_log = respath+'/results_'+label+'_'+audio_type+'_'+ker+str(d)+'c'+str(c)+'smotetest.log'
    f = open(result_log, 'w+')
    f.write('Results SMOTE\n%s Database, Features Compare2016 %ifold, audiotype and gender: %s\n' % (label, kfold, audio_type))
    f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f \n' % (ker, d, c))
    f.close()

    score = np.zeros((13,kfold))
    score_oracle = np.zeros((13,kfold))
    scoresmote = np.zeros((13,kfold))
    scoresmote_oracle = np.zeros((13,kfold))
    scoresmote2 = np.zeros((13,kfold))
    scoresmote11 = np.zeros((13,kfold))
    scoresmote12 = np.zeros((13,kfold))
    scoresmote13 = np.zeros((13,kfold))


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
        try: label_csv = label.split('_Nomiss')[0] 
        except: label_csv = label
        
        train_labels = np.array(train_labels)
        if not os.path.exists('data/features/'+label): os.mkdir('data/features/'+label)
        trainpath = 'data/features/'+label+'/train_'+audio_type+'_fold'+str(k+1)+'.pkl'
        if os.path.exists(trainpath): 
            with open(trainpath,'rb') as fid:
                train_features = pickle.load(fid)
                print('Fold '+ str(k+1) + ' Train: ' + str(train_features.shape))
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
        testpath = 'data/features/'+label+'/test_'+audio_type+'_fold'+str(k+1)+'.pkl'
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


        # 3. Train and Test SVM classifier
        model = SVC(C=c,kernel=ker,degree=d,probability=True)
        model.fit(train_features, train_labels)
        out = model.predict(test_features)
        out_oracle = model.predict(train_features)

        score[:,k] = compute_score(model, test_labels, out, test_features)
        with open (respath + '/output_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
            lbl = ['HEALTH','PATH']
            for j in range(0,len(test_files)):
                f.write('%s %s %s\n' % (os.path.basename(test_files[j])[:-4], lbl[test_labels[j]], lbl[out[j]]))
        
        score_oracle[:,k] = compute_score(model, train_labels, out_oracle, train_features)
        counter = Counter(train_labels)
        toc = time.time()
        f = open(result_log, 'a')
        f.write('\nHEALTH: %i, PATH: %i\n' % (counter[0], counter[1]))
        f.write('Oracle: Fold%i (%.2fsec): Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f, \n' % 
        (k+1, toc-tic, score_oracle[0,k], score_oracle[1,k], score_oracle[2,k], score_oracle[3,k], score_oracle[4,k], score_oracle[5,k], score_oracle[6,k], score_oracle[7,k], score_oracle[8,k], score_oracle[9,k], score_oracle[10,k], score_oracle[11,k], score_oracle[12,k]))
        f.close()        

        counter = Counter(train_labels)
        toc = time.time()
        f = open(result_log, 'a')
        f.write('Test Fold%i (%.2fsec): Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f, \n' % 
        (k+1, toc-tic, score[0,k], score[1,k], score[2,k], score[3,k], score[4,k], score[5,k], score[6,k], score[7,k], score[8,k], score[9,k], score[10,k], score[11,k], score[12,k]))
        f.close()

        # 4. Train and Test using SMOTE
        # SMOTE v1: Balance data
        train_features1, train_labels1 = augment.augment_smote_balance(train_features, train_labels, debug=False)
        model = SVC(C=c,kernel=ker,degree=d,probability=True)
        model.fit(train_features1, train_labels1)
        outsmote = model.predict(test_features)
        scoresmote[:,k] = compute_score(model, test_labels, outsmote, test_features)
        with open (respath + '/output_smotebalance_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
            lbl = ['HEALTH','PATH']
            for j in range(0,len(test_files)):
                f.write('%s %s %s\n' % (os.path.basename(test_files[j])[:-4], lbl[test_labels[j]], lbl[outsmote[j]]))
        
        counter = Counter(train_labels1)
        f = open(result_log, 'a')
        f.write('SMOTE_balance\n')
        f.write('HEALTH: %i, PATH: %i\n' % (counter[0], counter[1]))
        f.write('Fold%i (%.2fsec): Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f, \n' % 
        (k+1, toc-tic, scoresmote[0,k], scoresmote[1,k], scoresmote[2,k], scoresmote[3,k], scoresmote[4,k], scoresmote[5,k], scoresmote[6,k], scoresmote[7,k], scoresmote[8,k], scoresmote[9,k], scoresmote[10,k], scoresmote[11,k], scoresmote[12,k]))
        f.close()

        del train_features1
        del train_labels1
        gc.collect()

        # BorderlineSMOTE v1.1: Balance data 
        train_features11, train_labels11 = augment.augment_borderlinesmote_balance(train_features, train_labels)
        model = SVC(C=c,kernel=ker,degree=d,probability=True)
        model.fit(train_features11, train_labels11)
        outsmote = model.predict(test_features)
        scoresmote11[:,k] = compute_score(model, test_labels, outsmote, test_features)
        with open (respath + '/output_borderlinesmotebalance_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
            lbl = ['HEALTH','PATH']
            for j in range(0,len(test_files)):
                f.write('%s %s %s\n' % (os.path.basename(test_files[j])[:-4], lbl[test_labels[j]], lbl[outsmote[j]]))
        
        outsmoteoracle = model.predict(train_features11)
        scoresmote_oracle[:,k] = compute_score(model, train_labels11, outsmoteoracle, train_features11)
        counter = Counter(train_labels11)
        f = open(result_log, 'a')
        f.write('SMOTEORACLE_balance\n')
        f.write('Health: %i, Path: %i\n' % (counter[0], counter[1]))
        f.write('Fold%i (%.2fsec): Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f, \n' % 
        (k+1, toc-tic, scoresmote_oracle[0,k], scoresmote_oracle[1,k], scoresmote_oracle[2,k], scoresmote_oracle[3,k], scoresmote_oracle[4,k], scoresmote_oracle[5,k], scoresmote_oracle[6,k], scoresmote_oracle[7,k], scoresmote_oracle[8,k], scoresmote_oracle[9,k], scoresmote_oracle[10,k], scoresmote_oracle[11,k], scoresmote_oracle[12,k]))
        f.close()

        counter = Counter(train_labels11)
        f = open(result_log, 'a')
        f.write('BorderlineSMOTE_balance\n')
        f.write('HEALTH: %i, PATH: %i\n' % (counter[0], counter[1]))
        f.write('Fold%i (%.2fsec): Acc=%0.4f, AccHealth=%0.2f, AccPath=%0.2f, UAR=%0.4f, F1Score=%0.2f, Recall=%0.2f, Precision=%0.2f, AUC=%0.4f, EER=%0.4f, TP=%0.2f, TN=%0.2f, FP=%0.2f, FN=%0.2f, \n' % 
        (k+1, toc-tic, scoresmote11[0,k], scoresmote11[1,k], scoresmote11[2,k], scoresmote11[3,k], scoresmote11[4,k], scoresmote11[5,k], scoresmote11[6,k], scoresmote11[7,k], scoresmote11[8,k], scoresmote11[9,k], scoresmote11[10,k], scoresmote11[11,k], scoresmote11[12,k]))
        f.close()

        del train_features11
        del train_labels11
        gc.collect()


        # SVMSMOTE v1.2: Balance data 
        train_features12, train_labels12 = augment.augment_borderlinesmote_balance(train_features, train_labels)
        model = SVC(C=c,kernel=ker,degree=d,probability=True)
        model.fit(train_features12, train_labels12)
        outsmote = model.predict(test_features)
        scoresmote12[:,k] = compute_score(model, test_labels, outsmote, test_features)
        with open (respath + '/output_svmsmotebalance_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
            lbl = ['HEALTH','PATH']
            for j in range(0,len(test_files)):
                f.write('%s %s %s\n' % (os.path.basename(test_files[j])[:-4], lbl[test_labels[j]], lbl[outsmote[j]]))
        
        counter = Counter(train_labels12)
        f = open(result_log, 'a')
        f.write('SVMSMOTE_balance\n')
        f.write('HEALTH: %i, PATH: %i\n' % (counter[0], counter[1]))
        f.write('Fold%i (%.2fsec): Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f, \n' % 
        (k+1, toc-tic, scoresmote12[0,k], scoresmote12[1,k], scoresmote12[2,k], scoresmote12[3,k], scoresmote12[4,k], scoresmote12[5,k], scoresmote12[6,k], scoresmote12[7,k], scoresmote12[8,k], scoresmote12[9,k], scoresmote12[10,k], scoresmote12[11,k], scoresmote12[12,k]))
        f.close()

        del train_features12
        del train_labels12
        gc.collect()


        # ADASYNSMOTE v1.3: Balance data 
        train_features13, train_labels13 = augment.augment_borderlinesmote_balance(train_features, train_labels)
        model = SVC(C=c,kernel=ker,degree=d,probability=True)
        model.fit(train_features13, train_labels13)
        outsmote = model.predict(test_features)
        scoresmote13[:,k] = compute_score(model, test_labels, outsmote, test_features)
        with open (respath + '/output_adasynsmotebalance_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
            lbl = ['HEALTH','PATH']
            for j in range(0,len(test_files)):
                f.write('%s %s %s\n' % (os.path.basename(test_files[j])[:-4], lbl[test_labels[j]], lbl[outsmote[j]]))
        
        counter = Counter(train_labels13)
        f = open(result_log, 'a')
        f.write('ADASYNSMOTE_balance\n')
        f.write('HEALTH: %i, PATH: %i\n' % (counter[0], counter[1]))
        f.write('Fold%i (%.2fsec): Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f, \n' % 
        (k+1, toc-tic, scoresmote13[0,k], scoresmote13[1,k], scoresmote13[2,k], scoresmote13[3,k], scoresmote13[4,k], scoresmote13[5,k], 
        scoresmote13[6,k], scoresmote13[7,k], scoresmote13[8,k], scoresmote13[9,k], scoresmote13[10,k], scoresmote13[11,k], scoresmote13[12,k]))
        f.close()

        del train_features13
        del train_labels13
        gc.collect()

        '''
        # SMOTE v2: Multiply de dataset thorugh smote data augmentation 
        train_features2, train_labels2 = augment.augment_smote(train_features, train_labels, val=5, debug=False)
        model = SVC(C=c,kernel=ker,degree=d,probability=True)
        model.fit(train_features2, train_labels2)
        outsmote = model.predict(test_features)
        scoresmote2[:,k] = compute_score(model, test_labels, outsmote, test_features)
        with open (respath + '/output_smoteaugment_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
            lbl = ['HEALTH','PATH']
            for j in range(0,len(test_files)):
                f.write('%s %s %s\n' % (os.path.basename(test_files[j])[:-4], lbl[test_labels[j]], lbl[outsmote[j]]))
        
        counter = Counter(train_labels2)
        f = open(result_log, 'a')
        f.write('SMOTE_augment\n')
        f.write('HEALTH: %i, PATH: %i\n' % (counter[0], counter[1]))
        f.write('Fold%i (%.2fsec): Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f, \n' % 
        (k+1, toc-tic, scoresmote2[0,k], scoresmote2[1,k], scoresmote2[2,k], scoresmote2[3,k], scoresmote2[4,k], scoresmote2[5,k], scoresmote2[6,k], scoresmote2[7,k], scoresmote2[8,k], scoresmote2[9,k], scoresmote2[10,k], scoresmote2[11,k], scoresmote2[12,k]))
        f.close()

        del train_features2
        del train_labels2
        gc.collect()
        '''
        '''
        # SMOTE v3: Over and under sample
        counter = Counter(train_labels)
        print('HEALTH: %i, PATH: %i\n' % (counter[0], counter[1]))
        over = SMOTE(sampling_strategy=0.7)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [('over', over), ('under', under), ('model', model)]
        pipeline = Pipeline(steps=steps)
        pipeline.fit_resample(train_features, train_labels)
        counter = Counter(train_labels)
        print('HEALTH: %i, PATH: %i\n' % (counter[0], counter[1]))
        outsmote = pipeline.predict(test_features)
        '''

    # Mean     
    f = open(result_log, 'a')
    f.write('\nMean among folds\n')
    f.write('Oracle: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.mean(score_oracle[0,:]), np.mean(score_oracle[1,:]), np.mean(score_oracle[2,:]), np.mean(score_oracle[3,:]), np.mean(score_oracle[4,:]), np.mean(score_oracle[5,:]), 
    np.mean(score_oracle[6,:]), np.mean(score_oracle[7,:]), np.mean(score_oracle[8,:]), np.mean(score_oracle[9,:]), np.mean(score_oracle[10,:]), np.mean(score_oracle[11,:]), np.mean(score_oracle[12,:])))
    
    f.write('BASELINE: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.mean(score[0,:]), np.mean(score[1,:]), np.mean(score[2,:]), np.mean(score[3,:]), np.mean(score[4,:]), np.mean(score[5,:]), 
     np.mean(score[6,:]), np.mean(score[7,:]), np.mean(score[8,:]), np.mean(score[9,:]), np.mean(score[10,:]), np.mean(score[11,:]), np.mean(score[12,:])))
    
    f.write('SMOTE BAL: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.mean(scoresmote[0,:]), np.mean(scoresmote[1,:]), np.mean(scoresmote[2,:]), np.mean(scoresmote[3,:]), np.mean(scoresmote[4,:]), np.mean(scoresmote[5,:]), 
     np.mean(scoresmote[6,:]), np.mean(scoresmote[7,:]), np.mean(scoresmote[8,:]), np.mean(scoresmote[9,:]), np.mean(scoresmote[10,:]), np.mean(scoresmote[11,:]), np.mean(scoresmote[12,:])))
    
    f.write('BorderlineSMOTE BAL: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.mean(scoresmote11[0,:]), np.mean(scoresmote11[1,:]), np.mean(scoresmote11[2,:]), np.mean(scoresmote11[3,:]), np.mean(scoresmote11[4,:]), np.mean(scoresmote11[5,:]), 
     np.mean(scoresmote11[6,:]), np.mean(scoresmote11[7,:]), np.mean(scoresmote11[8,:]), np.mean(scoresmote11[9,:]), np.mean(scoresmote11[10,:]), np.mean(scoresmote11[11,:]), np.mean(scoresmote11[12,:])))
    
    f.write('SMOTEORACLE BAL: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.mean(scoresmote_oracle[0,:]), np.mean(scoresmote_oracle[1,:]), np.mean(scoresmote_oracle[2,:]), np.mean(scoresmote_oracle[3,:]), np.mean(scoresmote_oracle[4,:]), np.mean(scoresmote_oracle[5,:]), 
     np.mean(scoresmote_oracle[6,:]), np.mean(scoresmote_oracle[7,:]), np.mean(scoresmote_oracle[8,:]), np.mean(scoresmote_oracle[9,:]), np.mean(scoresmote_oracle[10,:]), np.mean(scoresmote_oracle[11,:]), np.mean(scoresmote_oracle[12,:])))
    
    f.write('SVMSMOTE BAL: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.mean(scoresmote12[0,:]), np.mean(scoresmote12[1,:]), np.mean(scoresmote12[2,:]), np.mean(scoresmote12[3,:]), np.mean(scoresmote12[4,:]), np.mean(scoresmote12[5,:]), 
     np.mean(scoresmote12[6,:]), np.mean(scoresmote12[7,:]), np.mean(scoresmote12[8,:]), np.mean(scoresmote12[9,:]), np.mean(scoresmote12[10,:]), np.mean(scoresmote12[11,:]), np.mean(scoresmote12[12,:])))
    
    f.write('ADASYNSMOTE BAL: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.mean(scoresmote13[0,:]), np.mean(scoresmote13[1,:]), np.mean(scoresmote13[2,:]), np.mean(scoresmote13[3,:]), np.mean(scoresmote13[4,:]), np.mean(scoresmote13[5,:]), 
     np.mean(scoresmote13[6,:]), np.mean(scoresmote13[7,:]), np.mean(scoresmote13[8,:]), np.mean(scoresmote13[9,:]), np.mean(scoresmote13[10,:]), np.mean(scoresmote13[11,:]), np.mean(scoresmote13[12,:])))
    
    
    '''
    f.write('SMOTE AUG: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f, \n' % 
    (np.mean(scoresmote[0,:]), np.mean(scoresmote[1,:]), np.mean(scoresmote[2,:]), np.mean(scoresmote[3,:]), np.mean(scoresmote[4,:]), np.mean(scoresmote[5,:]), 
     np.mean(scoresmote[6,:]), np.mean(scoresmote[7,:]), np.mean(scoresmote[8,:]), np.mean(scoresmote[9,:]), np.mean(scoresmote[10,:]), np.mean(scoresmote[11,:]), np.mean(scoresmote[12,:])))
    
    '''


    # Variance
    f.write('\nStandard deviation among folds\n')
    f.write('Oracle: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.std(score_oracle[0,:]), np.std(score_oracle[1,:]), np.std(score_oracle[2,:]), np.std(score_oracle[3,:]), np.std(score_oracle[4,:]), np.std(score_oracle[5,:]), 
    np.std(score_oracle[6,:]), np.std(score_oracle[7,:]), np.std(score_oracle[8,:]), np.std(score_oracle[9,:]), np.std(score_oracle[10,:]), np.std(score_oracle[11,:]), np.std(score_oracle[12,:])))
    
    f.write('BASELINE: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.std(score[0,:]), np.std(score[1,:]), np.std(score[2,:]), np.std(score[3,:]), np.std(score[4,:]), np.std(score[5,:]), 
     np.std(score[6,:]), np.std(score[7,:]), np.std(score[8,:]), np.std(score[9,:]), np.std(score[10,:]), np.std(score[11,:]), np.std(score[12,:])))
    
    f.write('SMOTE BAL: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.std(scoresmote[0,:]), np.std(scoresmote[1,:]), np.std(scoresmote[2,:]), np.std(scoresmote[3,:]), np.std(scoresmote[4,:]), np.std(scoresmote[5,:]), 
     np.std(scoresmote[6,:]), np.std(scoresmote[7,:]), np.std(scoresmote[8,:]), np.std(scoresmote[9,:]), np.std(scoresmote[10,:]), np.std(scoresmote[11,:]), np.std(scoresmote[12,:])))
    
    f.write('BorderlineSMOTE BAL: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.std(scoresmote11[0,:]), np.std(scoresmote11[1,:]), np.std(scoresmote11[2,:]), np.std(scoresmote11[3,:]), np.std(scoresmote11[4,:]), np.std(scoresmote11[5,:]), 
     np.std(scoresmote11[6,:]), np.std(scoresmote11[7,:]), np.std(scoresmote11[8,:]), np.std(scoresmote11[9,:]), np.std(scoresmote11[10,:]), np.std(scoresmote11[11,:]), np.std(scoresmote11[12,:])))
    
    f.write('SMOTEORACLE BAL: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.std(scoresmote_oracle[0,:]), np.std(scoresmote_oracle[1,:]), np.std(scoresmote_oracle[2,:]), np.std(scoresmote_oracle[3,:]), np.std(scoresmote_oracle[4,:]), np.std(scoresmote_oracle[5,:]), 
     np.std(scoresmote_oracle[6,:]), np.std(scoresmote_oracle[7,:]), np.std(scoresmote_oracle[8,:]), np.std(scoresmote_oracle[9,:]), np.std(scoresmote_oracle[10,:]), np.std(scoresmote_oracle[11,:]), np.std(scoresmote_oracle[12,:])))
    
    f.write('SVMSMOTE BAL: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.std(scoresmote12[0,:]), np.std(scoresmote12[1,:]), np.std(scoresmote12[2,:]), np.std(scoresmote12[3,:]), np.std(scoresmote12[4,:]), np.std(scoresmote12[5,:]), 
     np.std(scoresmote12[6,:]), np.std(scoresmote12[7,:]), np.std(scoresmote12[8,:]), np.std(scoresmote12[9,:]), np.std(scoresmote12[10,:]), np.std(scoresmote12[11,:]), np.std(scoresmote12[12,:])))
    
    f.write('ADASYNSMOTE BAL: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f \n' % 
    (np.std(scoresmote13[0,:]), np.std(scoresmote13[1,:]), np.std(scoresmote13[2,:]), np.std(scoresmote13[3,:]), np.std(scoresmote13[4,:]), np.std(scoresmote13[5,:]), 
     np.std(scoresmote13[6,:]), np.std(scoresmote13[7,:]), np.std(scoresmote13[8,:]), np.std(scoresmote13[9,:]), np.std(scoresmote13[10,:]), np.std(scoresmote13[11,:]), np.std(scoresmote13[12,:])))
    
    '''
    f.write('SMOTE AUG: Acc=%0.4f, AccHealth=%0.4f, AccPath=%0.4f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f, \n' % 
    (np.std(scoresmote[0,:]), np.std(scoresmote[1,:]), np.std(scoresmote[2,:]), np.std(scoresmote[3,:]), np.std(scoresmote[4,:]), np.std(scoresmote[5,:]), 
     np.std(scoresmote[6,:]), np.std(scoresmote[7,:]), np.std(scoresmote[8,:]), np.std(scoresmote[9,:]), np.std(scoresmote[10,:]), np.std(scoresmote[11,:]), np.std(scoresmote[12,:])))
    '''
    f.close() 
      
                        


    




#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print('Usage: run_voicedisorder_smote.py list_path kfold audio_type')
        print('Example: python run_voicedisorder_smote.py data/lst 5 aiu_both')
    else:
        list_path = args[0]
        kfold = int(args[1])
        audio_type = args[2]
        main(list_path, kfold, audio_type)
