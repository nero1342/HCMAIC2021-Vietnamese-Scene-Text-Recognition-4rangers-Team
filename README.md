# HCMAIC2021-4RANGERS
First Prize in AI-Challenge 2021, Ho Chi Minh City

# Team members:
- Nguyễn E Rô
- Nguyễn Trường Hải
- Nguyễn Ngọc Anh Khoa
- Hà Đức Minh Thảo

# Method
- Three-stage approach
    - Detection: YoloV5
    - Recognition: SATRN (MMOCR)
    - Post-processing: 
        <!-- -  
        -  -->

# Installation
## Requirements
```
    pip install -r requirements.txt
    cd source/det/yolov5/
    pip install -qr requirements.txt
    pip install mmcv-full==1.4.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
    pip install mmdet==2.19.1
    cd ../../mmocr/
    pip install -qr requirements.txt
    pip install -v -e .
```

# 