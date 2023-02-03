'''
Description: Compute features from pensmile (Compare2016) features  https://audeering.github.io/opensmile-python/
             and save a csv file with results in data/csv/name_smile.csv

Copyright: Vivolab, 2022
contact: dribas@unizar.es
'''


import os, sys, glob
import opensmile
sys.path.append('src')

def main(audio_path, outpath):
    if audio_path.endswith('list'):
        files = [] # assume we read a list of filepath
        with open(audio_path, 'r') as f:
            for line in f.readlines():
                files.append(line.strip())
        f.close()
    else:
        files = glob.glob(os.path.join(audio_path,'*.wav')) # assume we read directory
    
    if not os.path.isdir(outpath): os.makedirs(outpath)

    i=0
    # opensmile only works with single channel wav files
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
        )
    for wav in files:
        output = outpath + '/' + os.path.basename(wav)[:-4]+'_smile.csv'
        if not os.path.exists(output):
            print(str(i+1) + '. Processing: ' + wav)
            smileparam = smile.process_file(wav)
            #smileparam.to_excel(outpath + '/' + os.path.basename(wav)[:-4]+'_smile.xlsx')
            smileparam.to_csv(output)
            i=i+1




#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print('Usage: run_features_opensmile.py audio_path out_path')
        print('   audio_path: Directory or list(ext=".list") for audio files')
        print(' Example Dir: python bin/run_features_opensmile.py data/audio/Saarbruecken data/features/Saarbruecken')
        print(' Example List: python bin/run_features_opensmile.py data/lst/Saarbruecken.wav.list data/features/Saarbruecken')
    else:
        audio_path = args[0]
        out_path = args[1]
        main(audio_path, out_path)
