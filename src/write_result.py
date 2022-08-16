import pickle
import os
import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--path', help='Path')
args = parser.parse_args() 
path = args.path 


paths = []
with open('%s/test_list.txt' % path, 'r') as f:
    for line in f.readlines():
        p = line.strip().split()[0]
        paths.append(p)

new_res_fold0 = {} 
with open('%s/res_fold0.pkl' % path, 'rb') as f:
    res = pickle.load(f)    
for i in tqdm(range(len(paths))):
    name = paths[i]#.split('/')[-1]
    new_res_fold0[name] = res[i]

    
new_res_fold1 = {} 
with open('%s/res_fold1.pkl' % path, 'rb') as f:
    res = pickle.load(f)    
for i in tqdm(range(len(paths))):
    name = paths[i]#.split('/')[-1]
    new_res_fold1[name] = res[i]

new_res_fold2 = {} 
with open('%s/res_fold2.pkl' % path, 'rb') as f:
    res = pickle.load(f)    
for i in tqdm(range(len(paths))):
    name = paths[i]#.split('/')[-1]
    new_res_fold2[name] = res[i]

new_res_fold3 = {} 
with open('%s/res_fold3.pkl' % path, 'rb') as f:
    res = pickle.load(f)    
for i in tqdm(range(len(paths))):
    name = paths[i]#.split('/')[-1]
    new_res_fold3[name] = res[i]

    
new_res_fold4 = {} 
with open('%s/res_fold4.pkl' % path, 'rb') as f:
    res = pickle.load(f)    
for i in tqdm(range(len(paths))):
    name = paths[i]#.split('/')[-1]
    new_res_fold4[name] = res[i]

paths = glob.glob('%s/submission/*' % path)
try:
    os.makedirs('%s/e2e_result' % path)
except:
    pass
for path in tqdm(paths, total=len(paths)):
    with open(path, 'r') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split(',')
            name = line[-1]#.split('/')[-1]
            line[-1] = new_res_fold4[name]['text']
            text_fold0 = '|' + new_res_fold0[name]['text'] + '###' + str(new_res_fold0[name]['score']) 
            text_fold1 = '|' + new_res_fold1[name]['text'] + '###' + str(new_res_fold1[name]['score']) 
            text_fold2 = '|' + new_res_fold2[name]['text'] + '###' + str(new_res_fold2[name]['score']) 
            text_fold3 = '|' + new_res_fold3[name]['text'] + '###' + str(new_res_fold3[name]['score']) 
            text_fold4 = '|' + new_res_fold4[name]['text'] + '###' + str(new_res_fold4[name]['score']) 
            lines.append(','.join(line[:-1]) + text_fold0 + text_fold1 + text_fold2 +text_fold3 + text_fold4)
    FOLDER_SUBMISSION = path.replace('submission', 'e2e_result')
    #os.makedirs(FOLDER_SUBMISSION)
    with open(FOLDER_SUBMISSION, 'w') as f:
        for line in lines:
            print(line, file=f)


