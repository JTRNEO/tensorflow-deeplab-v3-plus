import cv2
import random
import numpy as np
import os
from keras.preprocessing.image import img_to_array
from deeplabv3 import Deeplabv3plus
from tqdm import tqdm




image_size = 512




    
def predict():
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = Deeplabv3plus('./model12/')
    stride = 512

        
    #load the image
    image = cv2.imread('./003_DG_Satellite_DXB_20180612.tif')
    print('image loaded')
    h,w,_ = image.shape
    padding_h = (h//stride + 1) * stride 
    padding_w = (w//stride + 1) * stride
    padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
    padding_img[0:h,0:w,:] = image[:,:,:]
    
    padding_img = img_to_array(padding_img)
    print ('src:',padding_img.shape)
    mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
    for i in tqdm(range(padding_h//stride)):
        for j in tqdm(range(padding_w//stride)):
            crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:]

            ch,cw,_ = crop.shape
            if ch != 512 or cw != 512:
                print ('invalid size!')
                continue
                
            #crop = np.expand_dims(crop, axis=0)
            cv2.imwrite('./patch/hehe.png',crop)
            #print 'crop:',crop.shape
            img_files=['./patch/hehe.png']
            pred = model.predict(img_files)
            for item in pred:
                img=item['decoded_labels']
                
                mask=img[:,:,0] 
            #print (np.unique(pred))  
                mask = mask.reshape((512,512)).astype(np.uint8)
                #print 'pred:',pred.shape
                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = mask[:,:]

    
    cv2.imwrite('./pre03.png',mask_whole[0:h,0:w])
        
    

    
if __name__ == '__main__':
    
    predict()



