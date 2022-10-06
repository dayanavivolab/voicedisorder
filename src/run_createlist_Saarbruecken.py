import os, sys
import numpy as np
from scipy.io import wavfile

'''
Description: Executable for creating list in json format for Saarbruecken database (binary clasification)
using same partitions in [Huckvale et al. Interspeech 2021] 
If audio_types = aiu, the utterances of a,i,u files are concatenated and saved in a new wavfile

Copyright: Vivolab, 2022
contact: dribas@unizar.es
'''

def main(audio_types, genders, partition_path, path_origen, path_destino, new_audios_path):
    
    print('Creating list \n Gender: %s \n Audio type: %s' % (genders, audio_types))

    if not os.path.exists(new_audios_path):
        temp_paths = ['HEALTH', 'PATH']
        os.mkdir(new_audios_path)
        for path in temp_paths:
            os.mkdir(os.path.join(new_audios_path, path))
            os.mkdir(os.path.join(new_audios_path, path, 'male'))
            os.mkdir(os.path.join(new_audios_path, path, 'female'))

    genders_list = ["male", "female"]
    types_required = []
    genders_required = []
    if audio_types == "all": types_required = ["phrase", "a_l", "a_n", "a_h", "a_lhl", "i_l", "i_n", "i_h", "i_lhl", "u_l", "u_n", "u_h", "u_lhl"]
    elif audio_types == "phrase": types_required = ["phrase"]
    elif audio_types == "lhl": types_required = ["a_lhl", "i_lhl", "u_lhl"]
    elif audio_types == "aiu": types_required = ["a_l", "a_n", "a_h", "a_lhl", "i_l", "i_n", "i_h", "i_lhl", "u_l", "u_n", "u_h", "u_lhl"]
    elif audio_types == "a": types_required = ["a_l", "a_n", "a_h", "a_lhl"]
    elif audio_types == "i": types_required = ["i_l", "i_n", "i_h", "i_lhl"]
    elif audio_types == "u": types_required = ["u_l", "u_n", "u_h", "u_lhl"]
    elif audio_types == "aiu_n": types_required = ["a_n", "i_n", "u_n"]
    elif audio_types == "a_n": types_required = ["a_n"]
    elif audio_types == "i_n": types_required = ["i_n"]
    elif audio_types == "u_n": types_required = ["u_n"]
    elif audio_types == "aiu_nlh": types_required = ["a_l", "a_n", "a_h", "i_l", "i_n", "i_h", "u_l", "u_n", "u_h"]
    else:
        print("audio type " + audio_types + " there is not this audio type")
        exit()

    if genders == "both": genders_required = genders_list
    else:
        if genders in genders_list: genders_required.append(genders)
        else:
            print("gender " + genders + " there is not this gender category")
            exit()

    kfold = len(os.listdir(partition_path))
    fold_numbers = []
    for file_name in os.listdir(partition_path):
        f = open(os.path.join(partition_path, file_name))
        fold_numbers.append([int(line.split(',')[0]) for line in f.readlines()])
        
    path_origen = os.path.join(path_origen)
    train = []
    test = []
    for k in range(kfold):
        train.append('{"labels": {"HEALTH": 0, "PATH": 1}, "meta_data": [')
        test.append('{"labels": {"HEALTH": 0, "PATH": 1}, "meta_data": [')
    
    labels_list = ["HEALTH", "PATH"]
    hombres, mujeres, norm, path = np.zeros((2,kfold)), np.zeros((2,kfold)), np.zeros((2,kfold)), np.zeros((2,kfold))
    k = 0
    for label in labels_list:
        for gender in genders_required:
            path_list = []
            if len(types_required) > 1:
                audio_list = os.listdir(os.path.join(path_origen, label, gender))
                speaker_list = []
                for audio in audio_list:
                    speaker = audio.split("-")[0]
                    if speaker not in speaker_list: speaker_list.append(speaker)

                for speaker in speaker_list:
                    required_audios = []
                    for audio_type in types_required:
                        audio_name = os.path.join(path_origen, label, gender, speaker + "-" + audio_type + ".wav")
                        try:
                            fs, wav = wavfile.read(audio_name)
                            required_audios.append(wav)
                        except: print(audio_name + " doesn't exist")
                    if len(required_audios) != 0:
                        new_audio = np.concatenate(required_audios)
                        path_audio = os.path.join(new_audios_path, label, gender, speaker + "-" + audio_types + ".wav")
                        path_list.append(path_audio)
                        wavfile.write(path_audio, fs, new_audio)

            else:
                for audio in os.listdir(os.path.join(path_origen, label, gender)):
                    temp = audio.split("-")
                    speaker = temp[0]
                    audio_type = temp[1].split(".")[0]
                    if audio_type in types_required:
                        path_audio = os.path.join(path_origen, label, gender, audio)
                        path_list.append(path_audio)


            for path_audio in path_list:
                speaker = os.path.basename(path_audio).split("-")[0]
                for i in range(0,kfold):                       
                    if int(speaker) in fold_numbers[i]:
                        if label == 'HEALTH': norm[0,i]+=1
                        if label == 'PATH': path[0,i]+=1
                        if gender == 'male': hombres[0,i]+=1
                        if gender == 'female': mujeres[0,i]+=1 
                        test[i] += '{"path": "' + path_audio + '", "label": "' + label + '", "speaker": "' + speaker + '"}, '

                    else:
                        for k in range(kfold):
                            if (k != i) and (int(speaker) in fold_numbers[k]):
                                if label == 'HEALTH': norm[1,i]+=1
                                if label == 'PATH': path[1,i]+=1
                                if gender == 'male': hombres[1,i]+=1
                                if gender == 'female': mujeres[1,i]+=1                            
                                train[i] += '{"path": "' + path_audio + '", "label": "' + label + '", "speaker": "' + speaker + '"}, '
                k+=1
                if k==kfold: k=0 

    if not os.path.exists(path_destino+"/log"): os.makedirs(path_destino+"/log")
    for k in range(0,kfold):
        # Save Test Json and Log
        test[k] = test[k][:-2]
        test[k] += ']}'
        f = open(path_destino+"/test_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write(test[k])
        f.close()
        f = open(path_destino+"/log/logtest_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write('Gender: male=%i, female=%i, total=%i\n' % (hombres[0,k], mujeres[0,k], hombres[0,k]+mujeres[0,k])) 
        f.write('Label: health=%i, path=%i, total=%i' % (norm[0,k], path[0,k], norm[0,k]+path[0,k]))
        f.close()        

        # Save Train Json and Log
        train[k] = train[k][:-2]
        train[k] += ']}'
        f = open(path_destino+"/train_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write(train[k])
        f.close()
        f = open(path_destino+"/log/logtrain_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write('Gender: male=%i, female=%i, total=%i\n' % (hombres[1,k], mujeres[1,k], hombres[1,k]+mujeres[1,k])) 
        f.write('Label: health=%i, path=%i, total=%i' % (norm[1,k], path[1,k], norm[1,k]+path[1,k]))
        f.close()




#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2:
        print('\nDescription: Create processing json list for Voice Disorder Detection system using Saarbruecken dataset\n')
        
        print('Usage: run_createlist_Saarbruecken.py audio_path label')
        print('  audio_types: Type of audio in Saarbruecken dataset: all | phrase | lhl | aiu | a | i | u | aiu_n | a_n | i_n | u_n ')
        print('  gender: gender of the speakers: male | female | both')
        print('  path_partition: path where folds are saved')
        print('  audio_path (optional): Directory for audio files (default: data/audio)')
        print('  json_path (optional): Output directory (default: data/lst)')
        print('  new audio path (optional): Directory of created audios for aiu concatenated (default: data/new_audio)\n')

        print('Example Dir: python run_createlist_Saarbruecken.py aiu both data/lst/saarbruecken_partitions/all_huckvale data/audio/Saarbruecken data/lst/Saarbruecken_concat data/audio/Saarbruecken/audios_concat  \n')
    
    else:
        audio_types = args[0]
        genders = args[1]
        partition = args[2]
        audio_path = 'data/audio'
        json_path = 'data/lst'
        new_audios_path = 'data/new_audio'
        if len(args)>3:
            audio_path = args[3]
        if len(args)>4:
            json_path = args[4]
        if len(args)>5:
            new_audios_path = args[5]
        
        main(audio_types, genders, partition, audio_path, json_path, new_audios_path)

