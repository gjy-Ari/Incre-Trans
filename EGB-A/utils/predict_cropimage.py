import os
import cv2
import math
import numpy as np

def Cropimage(image_paths,image_shape):
    out_dir = image_paths.split('.')[0]
    outimg_dir = os.path.join(out_dir,'Crop_images_{}'.format(image_shape[0]))
    if not os.path.exists(outimg_dir):
        os.makedirs(outimg_dir)
    ## *** Clip the img to x00*x00 ***  
    
    img = cv2.imread(image_paths)
    img = cv2.resize(img, (int(img.shape[1]*1), int(img.shape[0]*1)))
    pad_h = (math.ceil(img.shape[0]/image_shape[0]))*image_shape[0]
    pad_w = (math.ceil(img.shape[1]/image_shape[1]))*image_shape[1]
    pad_img = np.zeros((pad_h,pad_w,3),dtype='uint8')
    pad_img[:img.shape[0],:img.shape[1],:]=img
    image = pad_img
    
    row = image.shape[0]
    col = image.shape[1]
    for i in range(int(row/image_shape[0])):
        for j in range(int(col/image_shape[1])):
            subimage = image[i*image_shape[0]:(i+1)*image_shape[0],j*image_shape[1]:(j+1)*image_shape[1],:]
            cv2.imwrite(outimg_dir+'/'+os.path.basename(image_paths).split('.')[0]+'_'+'%02d'%(i+1)+'_'+'%02d'%(j+1)+'.tif',subimage)      
    return outimg_dir