import sys, os, pickle
sys.path.append('src')

import numpy as np
from scipy.io import savemat

'''
Description: Executable for computing a mat file with all features in pickle format

Copyright: Vivolab, 2022
contact: dribas@unizar.es
'''

def main(list_path, kfold, audio_type):

    label = os.path.basename(list_path)

    for k in range(0,kfold):        
        # Load features: Train
        trainpath = 'data/features/'+label+'/train_'+audio_type+'_fold'+str(k+1)+'.pkl'
        if os.path.exists(trainpath): 
            with open(trainpath,'rb') as fid:
                train_features = pickle.load(fid)
                ff = np.array(train_features, np.float)
                savemat(trainpath.replace('.pkl','.mat'),{'data': ff})
                print('Fold '+ str(k+1) +' Train: ' + str(train_features.shape))
        else:
            print('Error: pkl not exist')
        

        # Test
        testpath = 'data/features/'+label+'/test_'+audio_type+'_fold'+str(k+1)+'.pkl'
        if os.path.exists(testpath): 
            with open(testpath,'rb') as fid:
                test_features = pickle.load(fid)
                ff = np.array(test_features, np.float)
                savemat(testpath.replace('.pkl','.mat'),{'data': ff})
                print('Fold '+ str(k+1) + ' Test: ' + str(test_features.shape))
        else:
            print('Error: pkl not exist')           
        
          




#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print('audio_type (SVD): a_n, aiu, phrase')
        print('audio_type (AVFAD): aiu, phrase, read, spontaneous')
        print('Usage: run_features_pkl2mat.py list_path kfold audio_type')
        print('Example: python run_features_pkl2mat.py data/lst 5 phrase_both')
    else:
        list_path = args[0]
        kfold = int(args[1])
        audio_type = args[2]
        main(list_path, kfold, audio_type)
