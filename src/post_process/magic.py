import os 
from tqdm import tqdm
from pprint import pprint 
import numpy as np 
from shapely.geometry import * 

vnchar = 'aăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ1234567890-=qwertyuiop[]asdfghjkl;\'zxcvbnm,./"@!#%^&()+:?'
special_vowel_char = 'ăâáắấàằầảẳẩãẵẫạặậđêéếèềẻểẽễẹệíìỉĩịôơóốớòồờỏổởõỗỡọộợưúứùừủửũữụựýỳỷỹỵ'

specialCharacters=str(r'!?.:,*"()·[]/\'')
            
dict_path = './source/post_process/vn_dict.txt'
vn_dict = set() 
print("Learn Dictionary...")
with open(dict_path,'r') as f:
    lines = f.readlines() 
    for line in tqdm(lines):
        line = line.strip().upper()

        for label in line.split():
            vn_dict.add(label)
    
def accepted(label):
    pass 
    label2 = label[:]
    while(len(label2) and label2[0] in specialCharacters): label2 = label2[1:]
    while(len(label2) and label2[-1] in specialCharacters): label2 = label2[:-1]
    SUM_CHAR  = [c.isalpha() for c in label2].count(True)
    SUM_NUM = [c.isdigit() for c in label2].count(True)
    return label2 in vn_dict or (SUM_NUM > 0 and SUM_CHAR < 3)

def area(points):
    point = [
                [float(points[0]) , float(points[1])],
                [float(points[2]) , float(points[3])],
                [float(points[4]) , float(points[5])],
                [float(points[6]) , float(points[7])]
            ]
    pol = Polygon(point)
    return pol.area

def fix_ccw(points):
    point = [
            [float(points[0]) , float(points[1])],
            [float(points[2]) , float(points[3])],
            [float(points[4]) , float(points[5])],
            [float(points[6]) , float(points[7])]
        ]
    edge = [
            ( point[1][0] - point[0][0])*( point[1][1] + point[0][1]),
            ( point[2][0] - point[1][0])*( point[2][1] + point[1][1]),
            ( point[3][0] - point[2][0])*( point[3][1] + point[2][1]),
            ( point[0][0] - point[3][0])*( point[0][1] + point[3][1])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    pts = [] 
    if summatory>0:
        for i in range(3, -1, -1):
            pts.append(point[i][0])
            pts.append(point[i][1])
    else:
        pts = points
    return pts

import os, shutil

def make_archive(source, destination):
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s'%(name,format), destination)

def main(path, out_path):
    os.makedirs(out_path, exist_ok=True)
    lst = os.listdir(path)
    dict_labels = {}

    mapping_rules = {
        'THUY': 'THỤY',
        'HOÀ': 'HÒA',
        'DT:': 'ĐT:',
    }

    nonDict = total = fixed = 0 
    low_score = empty_label = small_bb = 0
    cnt_cw = cnt_unvalid = 0
    unique = 0
    totalx = 0
    import random 
    for f in tqdm(lst):
        fi = open(os.path.join(path, f), 'r') 
        lines = fi.readlines() 
        fo = open(os.path.join(out_path, f), 'w')
        for line in lines:
            l = line.strip().upper().split('|')
            pts = l[0].split(',')
            # Split 
            det_score = float(pts[-1])
            pts = pts[:8]
            labels = [x.split('###[') for x in l[1:]]
            labels = [x for x in labels if x[0] != ''] 
            scores = [np.array(list(map(float, x[1].strip()[0:-1].split(',')))) for x in labels]
            final_scores = [np.prod(score) for score in scores]
            labels = [x[0] for x in labels]
            
            
            # Find max freq
            totalx += 1
            maxFreq = max(set(labels), key=labels.count)
            if "<UKN>" in maxFreq: continue
            if labels.count(maxFreq) < len(labels) / 2: continue
            label = maxFreq
            
            if label not in dict_labels:
                dict_labels[label] = 0
            dict_labels[label] += 1 
            score = np.zeros(len(label))
            for i in range(len(labels)):
                if labels[i] == label:
                    try:
                        score += scores[i]
                    except:
                        print(f, score, scores[i], line)
            score /= labels.count(maxFreq)
            scores = score
            if label in mapping_rules:
                label = mapping_rules[label]

            if label == '':
                empty_label += 1
                continue 
            final_scores = np.prod(scores)

            if final_scores < 0.5:
                low_score += 1

            flag = True 
            points = list(map(float, pts))
            points2 = fix_ccw(points)
            if points != points2: cnt_cw += 1 
            points = fix_ccw(points2)
            if points != points2: 
                cnt_unvalid += 1 
                continue 
            pts = ','.join([str(point) for point in points])
            this_area = area(points)
            if this_area < 200:
                small_bb += 1
                
            # Condition
            if (not accepted(label) or final_scores < 0.5) and final_scores < 0.9:
                flag = False
            if flag:
                # CodaLab submission
                print(pts,label,sep = ',' ,file = fo, end = '\n') 
                
                # Self-evaluate
                # print(pts,label,sep = ',' ,file = fo, end = '|')
                # print(final_scores, *scores, file = fo) 
            total += 1 

    make_archive(out_path, out_path + '.zip')

    # print(nonDict, total)
    # print("Fixed", fixed)
    # print("Num low scores", low_score)
    # print("Empty label", empty_label)
    # print("Small bb", small_bb)
    # print("cnt_cw", cnt_cw)
    # print("cnt_unvalid", cnt_unvalid)
    # print(len(vn_dict))

    
from argparse import ArgumentParser
parser = ArgumentParser("Post process AIC")
parser.add_argument("--path", default="./submission_output")
parser.add_argument("--out", default="./final_submission_output")

args = parser.parse_args() 

main(args.path, args.out)

