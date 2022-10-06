import os, sys
import numpy as np
from scipy.io import wavfile
import pandas as pd

'''
Description: Executable for creating list in json format for AVFAD database (binary clasification)
If concat_flag = 1, the utterances of a,i,u files are concatenated and saved in a new wavfile

Copyright: Vivolab, 2022
contact: dribas@unizar.es
'''

concat_flag=0

def main(audio_types, genders, kfold, source_path, path_destino, new_audio_path, excel_metadata):
    
    print('Creating list \n Gender: %s \n Audio type: %s' % (genders, audio_types))

    if not os.path.exists(new_audio_path): os.mkdir(new_audio_path)
    if not os.path.exists(path_destino): os.mkdir(path_destino)
    if not os.path.exists(os.path.join(path_destino, 'log')): os.mkdir(os.path.join(path_destino, 'log'))

    if genders == 'both': genders_required = ['M', 'F']
    elif genders == 'male': genders_required = ['M']
    elif genders == 'female': genders_required = ['F']
    else:
        print("gender " + genders + " there is not this gender category")
        exit()

    if audio_types == 'all': types_required = np.arange(1, 12)
    elif audio_types == 'phrase': types_required = np.arange(4, 10)
    elif audio_types == 'i_n': types_required = np.arange(1, 2)
    elif audio_types == 'a_n': types_required = np.arange(2, 3)
    elif audio_types == 'u_n': types_required = np.arange(3, 4)
    elif audio_types == 'aiu': types_required = np.arange(1, 4)
    elif audio_types == 'read': types_required = np.arange(10, 11)
    elif audio_types == 'spontaneous': types_required = np.arange(11, 12)
    else:
        print("audio type " + audio_types + " there is not this audio type")
        exit()

    train = []
    test = []
    for k in range(kfold):
        train.append('{"labels": {"HEALTH": 0, "PATH": 1}, "meta_data": [')
        test.append('{"labels": {"HEALTH": 0, "PATH": 1}, "meta_data": [')
    male, female, health, path = np.zeros((2,kfold)), np.zeros((2,kfold)), np.zeros((2,kfold)), np.zeros((2,kfold))

    speakers_list = pd.read_excel(excel_metadata)
    k=0
    for _, speaker in speakers_list.iterrows():
        speaker_ID = speaker['File ID']
        label = speaker['CMVD-I Dimension 1 (numeric system)']
        if label == 0: label = 'HEALTH'
        else: label = 'PATH'
        gender = speaker['Sex']        
        if gender in genders_required:
            audio_list = []            
            for i in types_required:
                number = format(i, '03d')
                path_audio = os.path.join(source_path, speaker_ID, speaker_ID + number + '.wav')
                # Concatenate if phrase o aiu
                if (len(types_required)>1) and (concat_flag==1):
                    try:
                        fs, wav = wavfile.read(path_audio)
                        if wav.ndim == 2: wav = np.mean(wav, axis=1)
                        audio_list.append(wav)
                    except:
                        path_audio = os.path.join(source_path, speaker_ID, speaker_ID + number + '.WAV')
                        try:
                            fs, wav = wavfile.read(path_audio)
                            if wav.ndim == 2: wav = np.mean(wav, axis=1)
                            audio_list.append(wav)
                        except:
                            print(path_audio + " doesn't exist")
                    new_audio = np.concatenate(audio_list)
                    path_audio = os.path.join(new_audio_path, speaker_ID)
                    if not os.path.exists(path_audio): os.mkdir(path_audio)
                    path_audio = path_audio + '/' + speaker_ID + audio_types + '.wav'
                #    print(path_audio)
                    wavfile.write(path_audio, fs, new_audio)
                #else:
                #    print(path_audio)

                if os.path.exists(path_audio):
                    for i in range(0,kfold):
                        if i==k:
                            if label == 'HEALTH': health[0,i]+=1
                            if label == 'PATH': path[0,i]+=1
                            if gender == 'M': male[0,i]+=1
                            if gender == 'F': female[0,i]+=1 
                            test[i] += '{"path": "' + path_audio + '", "label": "' + label + '", "speaker": "' + speaker_ID + '"}, '
                            
                        else:
                            if label == 'HEALTH': health[1,i]+=1
                            if label == 'PATH': path[1,i]+=1    
                            if gender == 'M': male[1,i]+=1
                            if gender == 'F': female[1,i]+=1                      
                            train[i] += '{"path": "' + path_audio + '", "label": "' + label + '", "speaker": "' + speaker_ID + '"}, '
                            
                    k+=1
                    if k==kfold: k=0

    for k in range(0,kfold):
        # Save Test Json and Log
        test[k] = test[k][:-2]
        test[k] += ']}'
        f = open(path_destino+"/test_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write(test[k])
        f.close()
        f = open(path_destino+"/log/logtest_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write('Gender: male=%i, female=%i, total=%i\n' % (male[0,k], female[0,k], male[0,k]+female[0,k])) 
        f.write('Label: health=%i, path=%i, total=%i' % (health[0,k], path[0,k], health[0,k]+path[0,k]))
        f.close()        

        # Save Train Json and Log
        train[k] = train[k][:-2]
        train[k] += ']}'
        f = open(path_destino+"/train_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write(train[k])
        f.close()
        f = open(path_destino+"/log/logtrain_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write('Gender: male=%i, female=%i, total=%i\n' % (male[1,k], female[1,k], male[1,k]+female[1,k])) 
        f.write('Label: health=%i, path=%i, total=%i' % (health[1,k], path[1,k], health[1,k]+path[1,k]))
        f.close()
#---------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2:
        print('\nDescription: Create processing json list for s3prl baseline using AVFAD\n')
        print('Usage: run_createlist_AVFAD.py audio_path label')
        print(' audio_types: Type of audio in AVFAD dataset: all | phrase | aiu | read | spontaneous ')
        print(' gender: gender of the speakers: male | female | both')
        print(' new audio path (optional): Directory of created audios (default: data/audio/AVFAD/audio_16khz_concat)')
        print(' kfold: k number of lists for crossvalidation (default: 5)')
        print(' audio_path (optional): Directory for audio files (default: data/audio/AVFAD/audio_16khz)')
        print(' json_path (optional): Output directory (default: data/lst/AVFAD)')
        print(' excel with metadata (optional): Default data/audio/AVFAD/metadata.xlsx')
        print('Example Dir: python run_createlist_AVFAD.py phrase male\n')
    else:
        audio_types = args[0]
        genders = args[1]
        kfold = 5
        audio_path = 'data/audio/AVFAD/audio_16khz'
        json_path = 'data/lst/AVFAD'
        new_audio_path = 'data/audio/AVFAD/audio_16khz_concat'
        excel_metadata = 'data/audio/AVFAD/metadata.xlsx'

        if len(args)>2:
            new_audio_path = args[2]
        if len(args)>3:
            kfold = int(args[3])
        if len(args)>4:
            audio_path = args[4]
        if len(args)>5:
            json_path = args[5]
        if len(args)>6:
            excel_metadata = args[6]
        
        main(audio_types, genders, kfold, audio_path, json_path, new_audio_path, excel_metadata)
