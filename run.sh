#!/bin/sh

echo "Image Input:" $1;
echo "Detection Output: " $2;
echo "Recognition Output:" $3;
o=$2/exp

# rm -rf $3;
# ls weights/yolov5/
echo "Detecting..."
m_best="./weights/yolov5/yolov5m6_fold3_best.pt"
l_best="./weights/yolov5/yolov5l6_fold3_best.pt"
python3 src/det/yolov5/detect.py --weight $m_best $l_best --img 2304 --iou 0.15 --half --augment --conf-thres 0.4  --src $1 --save-txt --nosave --project $2 --exist-ok
# python3 src/det/yolov5/detect.py --weight $m_best --img 360 --iou 0.15 --half --augment --conf-thres 0.4  --src $1 --save-txt --nosave --project $2 --exist-ok

echo "Cropping images..."
python3 src/crop_images.py --label_path $o --data_path $1

rec_fold0='./weights/mmocr/fold0.pth'
rec_fold1='./weights/mmocr/fold1.pth'
rec_fold2='./weights/mmocr/fold2.pth'
rec_fold3='./weights/mmocr/fold3.pth'
rec_fold4='./weights/mmocr/fold4.pth'

cfgs='src/config/mmocr/starn.py'

echo "Recognizing..."
python3 src/mmocr/tools/test.py $cfgs $rec_fold0 $o --out $o/res_fold0.pkl
python3 src/mmocr/tools/test.py $cfgs $rec_fold1 $o --out $o/res_fold1.pkl
python3 src/mmocr/tools/test.py $cfgs $rec_fold2 $o --out $o/res_fold2.pkl
python3 src/mmocr/tools/test.py $cfgs $rec_fold3 $o --out $o/res_fold3.pkl
python3 src/mmocr/tools/test.py $cfgs $rec_fold4 $o --out $o/res_fold4.pkl

python3 src/write_result.py --path $o 

cp $o/e2e_result/* $3
echo "Post processing..."
python3 src/post_process/magic.py --path $3 --out $3