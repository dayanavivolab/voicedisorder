import sys, json, os, pickle
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.manifold import TSNE
#from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns


import augment
from utils import zscore

'''
Description: Dimension reduction with TSNE/UMAP projection to 2D of visualizing feature vectors 

Copyright: Vivolab, 2022
contact: dribas@unizar.es
'''


# Set Colors
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)


def main(list_path, kfold, audio_type):
    
    label = os.path.basename(list_path)

    # 1. Loading data from json list
    label_csv = label
    
    for k in range(0,kfold):
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


        # 3. Dimension reduction TSNE/UMAP 
        train_features_tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(train_features)
        #reducer = UMAP()
        #train_features_umap = reducer.fit_transform(train_features)
        
        plt.figure(1)
        sns.scatterplot(train_features_tsne[:,0], train_features_tsne[:,1], hue=train_labels, legend='full', palette=palette)
        plt.title('Original')
        plt.xlabel('TSNE dim 1')   
        plt.ylabel('TSNE dim 2') 
        plt.xlim((-60,60))
        plt.ylim((-60,60))
        plt.grid(True) 
        plt.savefig('data/figures/features-tsne-'+label+'-fold'+str(k+1)+'.png')
        plt.savefig('data/figures/features-tsne-'+label+'-fold'+str(k+1)+'.svg',format='svg',dpi=1000)
        plt.clf() 
        
        '''
        plt.figure(2)         
        plt.subplot(2,3,1)
        plt.scatter(train_features_umap[:,0], train_features_umap[:,1], c=[palette[i] for i in train_labels])
        plt.title('Original')
        plt.xlabel('UMAP dim 1')   
        plt.ylabel('UMAP dim 2') 
        plt.grid(True)
        '''

        # 4. SMOTE
        # SMOTE v1: Balance data
        train_features1, train_labels1 = augment.augment_smote_balance(train_features, train_labels, debug=False)
        train_features_tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(train_features1)
        #reducer = UMAP()
        #train_features_umap = reducer.fit_transform(train_features1)
        
        plt.figure(1)
        sns.scatterplot(train_features_tsne[:,0], train_features_tsne[:,1], hue=train_labels1, legend='full', palette=palette)
        plt.title('SMOTE')
        plt.xlabel('TSNE dim 1')   
        plt.ylabel('TSNE dim 2') 
        plt.xlim((-60,60))
        plt.ylim((-60,60))
        plt.grid(True) 
        plt.savefig('data/figures/features-tsne-'+label+'-SMOTE-fold'+str(k+1)+'.png')
        plt.savefig('data/figures/features-tsne-'+label+'-SMOTE-fold'+str(k+1)+'.svg',format='svg',dpi=1000)
        plt.clf() 

        '''
        plt.figure(2)         
        plt.subplot(2,3,2)
        plt.scatter(train_features_umap[:,0], train_features_umap[:,1], c=[palette[i] for i in train_labels1])
        plt.title('SMOTE')   
        plt.xlabel('UMAP dim 1')   
        plt.ylabel('UMAP dim 2') 
        plt.grid(True)
        '''

        # BorderlineSMOTE v1.1: Balance data 
        train_features11, train_labels11 = augment.augment_borderlinesmote_balance(train_features, train_labels)
        train_features_tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(train_features11)
        #reducer = UMAP()
        #train_features_umap = reducer.fit_transform(train_features11)
        
        plt.figure(1)
        sns.scatterplot(train_features_tsne[:,0], train_features_tsne[:,1], hue=train_labels11, legend='full', palette=palette)
        plt.title('Borderline SMOTE')
        plt.xlabel('TSNE dim 1')   
        plt.ylabel('TSNE dim 2') 
        plt.xlim((-60,60))
        plt.ylim((-60,60))
        plt.grid(True) 
        plt.savefig('data/figures/features-tsne-'+label+'-BorderlineSMOTE-fold'+str(k+1)+'.png')
        plt.savefig('data/figures/features-tsne-'+label+'-BorderlineSMOTE-fold'+str(k+1)+'.svg',format='svg',dpi=1000)
        plt.clf()   
        '''
        plt.figure(2)         
        plt.subplot(2,3,3)
        plt.scatter(train_features_umap[:,0], train_features_umap[:,1], c=[palette[i] for i in train_labels11])
        plt.title('Borderline SMOTE')   
        plt.xlabel('UMAP dim 1')   
        plt.ylabel('UMAP dim 2') 
        plt.grid(True)
        '''

        # SVMSMOTE v1.2: Balance data 
        train_features12, train_labels12 = augment.augment_borderlinesmote_balance(train_features, train_labels)
        train_features_tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(train_features12)
        #reducer = UMAP()
        #train_features_umap = reducer.fit_transform(train_features12)
        
        plt.figure(1)
        sns.scatterplot(train_features_tsne[:,0], train_features_tsne[:,1], hue=train_labels12, legend='full', palette=palette)
        plt.title('SVM SMOTE')
        plt.xlabel('TSNE dim 1')   
        plt.ylabel('TSNE dim 2') 
        plt.xlim((-60,60))
        plt.ylim((-60,60))
        plt.grid(True) 
        plt.savefig('data/figures/features-tsne-'+label+'-SVMSMOTE-fold'+str(k+1)+'.png')
        plt.savefig('data/figures/features-tsne-'+label+'-SVMSMOTE-fold'+str(k+1)+'.svg',format='svg',dpi=1000)
        plt.clf() 
        '''
        plt.figure(2)         
        plt.subplot(2,3,4)
        plt.scatter(train_features_umap[:,0], train_features_umap[:,1], c=[palette[i] for i in train_labels12])
        plt.title('SVM SMOTE')   
        plt.xlabel('UMAP dim 1')   
        plt.ylabel('UMAP dim 2') 
        plt.grid(True)
        '''

        # ADASYNSMOTE v1.3: Balance data 
        train_features13, train_labels13 = augment.augment_borderlinesmote_balance(train_features, train_labels)
        train_features_tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(train_features13)
        #reducer = UMAP()
        #train_features_umap = reducer.fit_transform(train_features13)
        
        plt.figure(1)
        sns.scatterplot(train_features_tsne[:,0], train_features_tsne[:,1], hue=train_labels13, legend='full', palette=palette)
        plt.title('ADASyn SMOTE')
        plt.xlabel('TSNE dim 1')   
        plt.ylabel('TSNE dim 2') 
        plt.xlim((-60,60))
        plt.ylim((-60,60))
        plt.grid(True) 
        plt.savefig('data/figures/features-tsne-'+label+'-ADASynSMOTE-fold'+str(k+1)+'.png')
        plt.savefig('data/figures/features-tsne-'+label+'-ADASynSMOTE-fold'+str(k+1)+'.svg',format='svg',dpi=1000)
        plt.clf() 
        '''
        plt.figure(2)         
        plt.subplot(2,3,5)
        plt.scatter(train_features_umap[:,0], train_features_umap[:,1], c=[palette[i] for i in train_labels13])
        plt.title('ADASyn SMOTE')   
        plt.xlabel('UMAP dim 1')   
        plt.ylabel('UMAP dim 2') 
        plt.grid(True)
        '''
        

#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print('Usage: run_visualize_features_tsne-umap.py list_path kfold audiotype_gender')
        print('Example: python sun_visualize_features_tsne-umap.py data/lst/Saarbruecken 5 phrase_both')
    else:
        list_path = args[0]
        kfold = int(args[1])
        audio_type = args[2]
        main(list_path, kfold, audio_type)


