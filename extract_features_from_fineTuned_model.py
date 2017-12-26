#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:50:24 2017

@author: raul
""" 
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import keras
from keras import backend as tf

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

from keras.applications.vgg16 import preprocess_input as ppi_vgg16
from keras.applications.vgg19 import preprocess_input as ppi_vgg19
from keras.applications.xception import preprocess_input as ppi_xception
from keras.applications.resnet50 import preprocess_input as ppi_resnet50
from keras.applications.inception_v3 import preprocess_input as ppi_inception_v3
from keras.applications.mobilenet import preprocess_input as ppi_mobilenet

from keras.preprocessing import image
from keras.layers import Input

from keras.models import model_from_json

np.set_printoptions(threshold=np.nan)
keras.backend.backend()

import time
current_milli_time = lambda: int(round(time.time() * 1000))

def load_model(model_path,modelName,pooltype):

    # load json and create model
    json_file = open(model_path+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path+"model.h5")
    print("Loaded "+modelName+" model from disk")
    
    if (modelName in ['InceptionV3','Xception']):
        targetSize = 299  
    else:
        targetSize = 224   
    
    return  model,targetSize

def extractDeepFeatures(filepath,model,modelName,target_size,axis_name,mid):
    img = load_img(filepath,target_size,axis_name,mid)    
    x = image.img_to_array(img)    
    x = np.expand_dims(x, axis=0)

    if (modelName is 'VGG16'):
        x = ppi_vgg16(x)        
    if (modelName is 'VGG19'):
        x = ppi_vgg19(x)
    if (modelName is 'ResNet50'):
        x = ppi_resnet50(x)
    if (modelName is 'MobileNet'):
        x = ppi_mobilenet(x)
    if (modelName is 'Xception'):
        x = ppi_xception(x) 
    if (modelName is 'InceptionV3'):
        x = ppi_inception_v3(x)

#     timeStart = current_milli_time()
    features = model.predict(x).reshape(-1)
#     print('Load process: Complete',(current_milli_time()-timeStart)/1000)

    
    return features

def load_img(filepath,target_size,axis_name,mid):
    
    """Load LIDC-IDRI dataset"""
    
    img = [] 
    #LOAD image to be deep extracted         
    if (filepath.endswith('jpg')):
        img = image.load_img(filepath)
    elif (filepath.endswith('txt')):
        img = np.loadtxt(filepath).copy()
    elif (filepath.endswith('npy')):
        img = np.load(filepath).copy()
        
    if (axis_name is '3Axis'):
        img1 = np.array(img[mid,:,:])
        img1 = cv2.resize(img1, (target_size,target_size), interpolation = cv2.INTER_NEAREST)

        img2 = np.array(img[:,mid,:])
        img2 = cv2.resize(img2, (target_size,target_size), interpolation = cv2.INTER_NEAREST)

        img3 = np.array(img[:,:,mid])
        img3 = cv2.resize(img3, (target_size,target_size), interpolation = cv2.INTER_NEAREST)

        img = np.array([img1,img2,img3]).T
        
        #     plt.imshow(img[:,:,0])
        #     plt.show()
        #     plt.imshow(img[:,:,1])
        #     plt.show()
        #     plt.imshow(img[:,:,2])
        #     plt.show()
    else:
        if (axis_name is 'Axial'):
            img = np.array(img[mid,:,:])
        if (axis_name is 'Coronal'):
            img = np.array(img[:,mid,:])
        if (axis_name is 'Sagittal'):
            img = np.array(img[:,:,mid])
        img = cv2.resize(img, (target_size,target_size), interpolation = cv2.INTER_NEAREST)
        img = np.array([img,img,img]).T
        # plt.imshow(img)
        # plt.show()

    return img


vol_size_list = [8,16,24,32,48,64]
vol_size_list = [32]


for vol_size in vol_size_list:
    ## Especifique o caminho da pasta onde estão as imagens a serem processadas.
    srcPath = '/home/raul/PROJECTS/LIDC-IDRI/Out_Folder_NODULES_NPY/Mean/'+str(vol_size)+'x'+str(vol_size)+'x'+str(vol_size)+'/' 

    ## Especifique o caminho da pasta onde serão armazenadas as caracteristicas extridas.
    outPath = './Out_Folder/LIDC-IDRI/Mean/'+str(vol_size)+'x'+str(vol_size)+'x'+str(vol_size)+'/' 

    #Especifique numero de classes do problema.
    numOfClasses = 5 
    
    #Especifique a topologia a ser utilizada.
    model_path_list = []
    model_name_list = ['VGG16','VGG19','MobileNet','ResNet50','InceptionV3','Xception']
#     model_name_list = ['InceptionV3','Xception']

    #Especifique o exito(s) do nodulo a serem utilizados.
    axis_name_list = ['3Axis','Coronal','Sagittal','Axial']
    axis_name_list = ['Axial']

    #Especifique o metodo de pooling a ser utilizado.
    # pooltype = ['max','avg']
    pooltype_list = ['max'] 

    for pooltype in pooltype_list:
        for axis_name in axis_name_list:
            outPathA = outPath+axis_name+'/'
            print('\n\n\n\n',outPathA)
            for model_name in model_name_list:
                # For each folder / class                    
                model,targetSize = load_model(model_name,pooltype)
                
                data = []
                for fileIdx in range(numOfClasses):

                    folderPath = srcPath+str(fileIdx)

                    for subdir, dirs, files in os.walk(folderPath):
                        for idx,name in enumerate(files):
                            filePath = subdir + os.sep + name
                            print(idx, filePath)

                            features = extractDeepFeatures(filePath,model,model_name,targetSize,axis_name,int(vol_size/2))
                            features = np.hstack((features,fileIdx))
                            data.append(features)

                # Creates a folder if don't exist
                os.makedirs(outPathA+ pooltype+'/', exist_ok=True) 
                np.savetxt(outPathA + pooltype +'/'+model_name+'.txt',data,fmt="%.8f")
                np.save(outPathA + pooltype +'/'+model_name+'.npy',data)

