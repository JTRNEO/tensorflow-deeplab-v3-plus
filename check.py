import numpy as np 
import json
import cv2
import os 
path='./SegmentationClass/'
a=os.listdir(path)
a.sort(key=lambda x:int(x[:-4]))
for i in a:
    img=cv2.imread(path+i,cv2.IMREAD_GRAYSCALE)
    print(set(img.flatten()))