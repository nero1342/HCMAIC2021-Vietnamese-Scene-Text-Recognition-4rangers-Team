import glob 
import cv2
import os
import tqdm
import argparse
parser = argparse.ArgumentParser(description='Crop images from yolov5 and write pseudo submission')
parser.add_argument('--label_path', help = 'Label from yolov5 path')
parser.add_argument('--data_path', help = 'Path to the inference data')
args = parser.parse_args()
# vinbrain/nthai/AIC2021/yolov5/runs/detect/testA_yolov5l6_2048_0.5_augment
NEW_IMAGES_PATH = os.path.join(args.label_path, 'crop')
TXT_SUBMISSION = os.path.join(args.label_path, 'submission')
txt_path = glob.glob('{}/*'.format(os.path.join(args.label_path, 'labels')))
try:
    os.makedirs(NEW_IMAGES_PATH)
    os.makedirs(TXT_SUBMISSION)
except FileExistsError:
    pass
id = 0
new_labels = []
k = 1
for txt in tqdm.tqdm(txt_path):
    txt_submission = os.path.join(TXT_SUBMISSION, txt.split('/')[-1]) #'{}/{}'.format(TXT_SUBMISSION, txt.split('/')[-1])
    
    with open(txt, 'r') as f:
        img_path = os.path.join(args.data_path,txt.split('/')[-1][:-4])#'{}/{}'.format(args.data_path,txt.split('/')[-1][:-4])
        img = cv2.imread(img_path)
        fi = open(txt_submission, 'a+')
        for line in f.readlines():
            x1, y1, x2, y2, x3, y3, x4, y4, conf, _ = line.split(',')
            x1, y1, x2, y2, x3, y3, x4, y4, conf = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4), float(conf)
            new_img_path = os.path.join(NEW_IMAGES_PATH, "word_" + str(id) + ".jpg")
            id += 1
            crop_x1, crop_y1, crop_x3, crop_y3 = x1, int(k * y1), x3, y3 #int((2 - k) * x3), int((2 - k) * y3)
            word = img[crop_y1: crop_y3, crop_x1 : crop_x3]
#             word = img[y1 : y3, x1 : x3]
            cv2.imwrite(new_img_path, word)
            new_img_path = '/'.join(new_img_path.split('/')[-2:])
            new_labels.append((new_img_path, '0'))
            fi.write('{},{},{},{},{},{},{},{},{},{}\n'.format(x1, y1, x2, y2, x3, y3, x4, y4, conf, new_img_path))
            
#             break
        fi.close()
        
#     break

import glob

paths = glob.glob('{}/*'.format(NEW_IMAGES_PATH))
with open(os.path.join(args.label_path, 'test_list.txt'), 'w+') as f:
    for path in paths:
        name = '/'.join(path.split('/')[-2:])
        print(name + ' label', file = f)
#         break
#         print(name, file=f)
            