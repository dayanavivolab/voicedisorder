import os, glob
import numpy as np

direc = 'data/result'

files = glob.glob(os.path.join(direc,'*.log'))

result_dict = {}
result_dict_1 = {}
init = 0
hit = {}
miss = {}
for fil in files:
    print(str(init) + '. ' + fil)
    with open(fil) as fid:
        res = fid.readlines()
    for line in res:
        name = line.split(' ')[0].split('-')[0]
        true = line.split(' ')[1]
        estim = line.split(' ')[2].split('\n')[0]
        if init == 0:
            result_dict[name] = [true, estim]
            if true == estim: 
                hit[name] = 1
                miss[name] = 0
            else: 
                miss[name] = 1
                hit[name] = 0
            result_dict_1[name] = [true, hit[name], miss[name]]
        else:
            try:
                result_dict[name].append(estim)
                if true == estim: hit[name]+=1
                else: miss[name]+=1
                result_dict_1[name] = [true, hit[name], miss[name]]
            except:
                continue
    init += 1

with open(direc + '/comparison.txt', 'w') as fid:
    fid.write('name:[true,estim,...]\n')
    for key, value in result_dict.items(): 
        fid.write('%s:%s\n' % (key, value))

with open(direc + '/comparison_values.txt', 'w') as fid:
    fid.write('name:[true,hit,miss]\n')
    for key, value in result_dict_1.items(): 
        fid.write('%s:%s\n' % (key, value))
        

    

        
        
