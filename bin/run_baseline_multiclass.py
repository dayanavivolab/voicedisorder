'''
Description: Executable for run the Voice Disorder Detection system (multi-class classification)
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
from sklearn.metrics import classification_report

from utils import compute_score_multiclass, zscore
from collections import Counter
import matplotlib.pyplot as plt

def main(list_path, kfold, audio_type):

    ker='poly'
    d=1
    c=1
    label = os.path.basename(list_path)

    result_log = 'results_'+label+'_'+audio_type+'_'+ker+str(d)+'c'+str(c)+'.log'
    f = open(result_log, 'w+')
    f.write('Results Data:%s Features:Compare2016 %ifold, %s\n' % (label, kfold, audio_type))
    f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f \n' % (ker, d, c))
    f.close()

    respath = 'data/result/'+label
    if not os.path.exists(respath): os.mkdir(respath)
    
    # 1. Loading data from json list
    for k in range(0,kfold):
        tic = time.time()
        train_files = [] 
        train_labels = [] 
        trainlist = list_path + '/train_' + audio_type + '_meta_data_fold' + str(k+1) + '.json'
        with open(trainlist, 'r') as f:
            data = json.load(f)
            #c = 0
            for item in data['meta_data']:
                train_files.append(item['path'])
                train_labels.append(item['label'])
                #c +=1
                #print(str(c) + ' ' + item['path'] + ' ' + str(item['label']))
        f.close()

        test_files = [] 
        test_labels = [] 
        testlist = list_path + '/test_' + audio_type + '_meta_data_fold' + str(k+1) + '.json'
        with open(testlist, 'r') as f:
            data = json.load(f)
            #c = 0
            for item in data['meta_data']:
                test_files.append(item['path'])
                test_labels.append(item['label'])
                #c +=1
                #print(str(c) + ' ' + item['path'] + ' ' + str(item['label']))
        f.close()


        # 2. Load features: Train
        audio_type_pkl = audio_type.split('multi_')[1]
        train_labels = np.array(train_labels)
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
                feat = pd.read_csv('data/features/'+label+'/'+name+'_smile.csv').to_numpy()[0]
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
                feat = pd.read_csv('data/features/'+label+'/'+name+'_smile.csv').to_numpy()[0]
                test_features.append(feat[3:])
                i=i+1
            print('Test: ' + str(i))
            test_features = np.array(test_features) 
            with open(testpath,'wb') as fid:
                pickle.dump(test_features, fid, protocol=pickle.HIGHEST_PROTOCOL)            
        test_features = zscore(test_features, trainmean, trainstd)


        # 3. Train SVM classifier
        counter = Counter(train_labels)
        labels = data['labels'].keys()
        sizes = []
        for i in labels:
            sizes.append(counter[data['labels'][i]])
        '''
        plt.subplot(121)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title('Train')  
        '''
        counter = Counter(test_labels)
        sizes = []
        for i in labels:
            sizes.append(counter[data['labels'][i]])
        '''
        plt.subplot(122)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title('Test')  
       # plt.show()
        '''
        clf = SVC(C=c,kernel=ker,degree=d)
        clf.fit(train_features, train_labels)

        # 4. Testing
        out = clf.predict(test_features)
        out_oracle = clf.predict(train_features)

        score = compute_score_multiclass(test_labels, out, data)
        score_oracle = compute_score_multiclass(train_labels, out_oracle, data)
        
        lbl = {}
        for i in labels:
            lbl[data['labels'][i]] = i
        with open (respath+'/output_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
            for j in range(0,len(test_files)):
                f.write('%s %s %s\n' % (os.path.basename(test_files[j])[:-4], lbl[test_labels[j]], lbl[out[j]]))
        
        with open (respath+'/output_oracle_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
            for j in range(0,len(train_files)):
                f.write('%s %s %s\n' % (os.path.basename(train_files[j])[:-4], lbl[train_labels[j]], lbl[out_oracle[j]]))
        
        toc = time.time()
        f = open(result_log, 'a')
        f.write('\nOracle Fold%i (%.2fsec)' % (k, toc-tic))
        f.write(score_oracle) 
        f.close()
        f = open(result_log, 'a')
        f.write('\nTest Fold%i (%.2fsec)' % (k, toc-tic))
        f.write(score) 
        f.close()
        

    





    




#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print('audio_type (SVD): multi_a_n, multi_aiu, multi_phrases')
        print('audio_type (AVFAD): multi_aiu, multi_phrases, multi_read, multi_spontaneous')
        print('Usage: run_baseline.py list_path kfold audio_type')
        print('Example: python run_baseline.py data/lst 5 phrase_both')
    else:
        list_path = args[0]
        kfold = int(args[1])
        audio_type = args[2]
        main(list_path, kfold, audio_type)
