#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:14:48 2017

LIDC-IDRI / CNN pipeline

@author: raul
"""
import os
import numpy as np
import random
import scipy
import cv2
import tensorflow as tf
import glob
import h5py

from sklearn.metrics import confusion_matrix
import itertools

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, MaxPooling2D , Flatten ,Input, Dropout
from keras import backend as K
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import optimizers

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
            
class LIDC_IDRI:
    def simplify_prob(x_data,y_data):
        new_y = [];new_x = [];
        for x,y in zip(x_data,y_data):
            if (y>3):
                new_y.append(1)
                new_x.append(x)
            if (y<3):
                new_y.append(0)
                new_x.append(x)
        return new_x,new_y,2
    
    def load_data(folder_path,target_size,multiclass=False,force_balance=False,shuffle=True,clone=False):
        
        x = []; y = []
        for classID in os.listdir(folder_path):
            path_classID = folder_path+'/'+classID                          
            for file_path in glob.glob(path_classID+"/*"):    
                img = np.load(file_path)
                img = cv2.resize(img, (target_size,target_size), interpolation = cv2.INTER_NEAREST)
                x.append(img)
                y.append(int(classID))
                  
        nClass = len(os.listdir(folder_path))
    
        if(multiclass == False):
            x,y,nClass = LIDC_IDRI.simplify_prob(x,y)
        if(shuffle == True):
            x,y = LIDC_IDRI.shuffle(x, y)

        return np.array(x),np.array(y),nClass
    

    def shuffle(x_data, y_data):
        c = list(zip(x_data, y_data))
        random.shuffle(c)
        return zip(*c)

def buildModel(modelName,input_tensor):
    if ('VGG16' is modelName):
        return VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)    
    if ('VGG19' is modelName):
        return VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
    if ('MobileNet' is modelName):
        return MobileNet(weights='imagenet', include_top=False, input_tensor=input_tensor)  
    if ('ResNet50' is modelName):
        return ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)       
    if ('InceptionV3' is modelName):
        return InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
    if ('Xception' is modelName):
        return Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)  
    
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def slice_nodule(imgs,labels,target_size,num_slices=5,stride=1):
    half_vol = imgs.shape[-1] / 2
    num_slices = int(num_slices / 2)                         
    res_imgs = []; res_labels = [];
    for vol, label in zip(imgs,labels):
        for img in vol.T[half_vol-num_slices:half_vol+num_slices+1:stride,:,:]:
            res_imgs.append(np.array([img,img,img]).T)
            res_labels.append(label)               
    return np.array(res_imgs),np.array(res_labels)


#Paths 32MedianAnnotation
srcPath = '/home/raul/PROJECTS/LIDC-IDRI/Out_Folder_PYTHON/16x16x16'

# class_weight = {0: 1 ,1: 5}
class_weight = None

# Resize source image
target_size_list = [224]

# Base model to be fine-tuned 
modelName_list = ['ResNet50']

# fine-tuning from layer
layer2Freeze_list = [10]

# number of epochs 
epochs_MLP = 5
epochs_CNN = 5

# Loading Variables
multiclass = False
shuffle = True

# number of samples in a batch
batch_size = 32

# number of folds
K_fold = 5

out_path = './outputs/'

output_path= out_path+srcPath.split('/')[-1]

for modelName,target_size,layer2Freeze in zip(modelName_list,target_size_list,layer2Freeze_list):
    
    ############################################
    output_name= str(np.random.randint(10000000))+'_'+modelName+'_'+str(target_size)+'_'+str(layer2Freeze)
    log_path = output_path+output_name+'/'
    print(output_name)

    os.makedirs(log_path, exist_ok=True)

    file = open(log_path+output_name+".txt","w") 
    file.write(output_name+'\n')

    ####
    srcFolder = srcPath+'/'
    ############################################

    #Load Ds    
    x_data,y_data,num_classes = LIDC_IDRI.load_data(srcFolder,target_size,multiclass,shuffle)

    #Onehot labels
    y_data_hot = np_utils.to_categorical(y_data, num_classes)

    print('Loaded DS has:', len(x_data), 'samples /',num_classes,'classes')
    file.write('Loaded DS has: '+str(len(x_data))+' samples | '+str(num_classes)+' classes\n') 

    ## Log Variables
    y_pred_Total = []
    y_true_Total = []

    ## print the number of images per class
    print([np.sum([1 for i in y_data if i == x]) for x in range(num_classes)])

    #############################################
    skf = StratifiedKFold(n_splits=K_fold)
    for idx , (train, test) in enumerate(skf.split(x_data, y_data)):
        #############################
        x_train_vol = np.array(x_data[train])
        y_train_vol = np.array(y_data_hot[train])
        x_test_vol  = np.array(x_data[test])
        y_test_vol  = np.array(y_data_hot[test])
        #############################
        print(x_train_vol.shape,y_train_vol.shape)
        x_train, y_train = slice_nodule(x_train_vol,y_train_vol,target_size,1)
        print(x_train.shape,y_train.shape)

        print(x_test_vol.shape,y_test_vol.shape)
        x_test , y_test = slice_nodule(x_test_vol,y_test_vol,target_size,1)
        print(x_test_vol.shape,y_test_vol.shape)
        
        x_train_vol = []
        y_train_vol = []
        x_test_vol = []
        y_test_vol = []
        #############################
        print('\n###########################################################################################')
        print('K-fold number: ',idx+1,'\n')
        file.write('\n\nK-fold number: '+str(idx+1)+'\n')

        # Create the base pre-trained model
        input_tensor = Input(shape=(target_size,target_size,3))
        base_model = buildModel(modelName,input_tensor)                
        base_model.summary()

        # Add a Flatten layer
        x = base_model.output
        x = Flatten()(x)
                      
##         Add a fully-connected layer
#        x = Dense(512, activation='relu')(x)
#        x = Dropout(0.8)(x)
#        x = Dense(128, activation='relu')(x)


        # make a new softmax layer with num_classes neurons
        predictions = Dense(num_classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        print('number of layer: ',len(model.layers))
 
        ##################################################################
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional base_model layers
#        for layer in base_model.layers:
#            layer.trainable = False
            
        for layer in model.layers[:-1]:
            layer.trainable = False
            
        # ensure the last layer is trainable/not frozen
        for layer in model.layers[-1:]:
            layer.trainable = True
        ##################################################################
        adam=optimizers.Adam(lr=1e-03, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=adam, loss='mean_squared_error',metrics=['accuracy'])

        # train the model on the new data for a few epochs
        datagen = ImageDataGenerator(
                horizontal_flip = True,
                vertical_flip = True)
        
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        # datagen.fit(x_train)

        # fits the model on batches with real-time data augmentation:
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=len(x_train) / batch_size, epochs=epochs_MLP,
                            verbose=1, validation_data=(x_test, y_test),
                            class_weight = class_weight)
        ###############################################################
        # METRICS
        print('\nModel Evaluation\n')
        file.write('\nModel Evaluation\n')
        y_hat = model.predict(x_test, batch_size=batch_size,verbose=1)
        y_pred = np.argmax(y_hat,axis=1)
        y_true = np.argmax(y_test,axis=1)
        print(classification_report(y_true, y_pred))
        file.write(classification_report(y_true, y_pred))
        file.write('\n==========================\n')
#       ###############################################################               
#        
##                 # we chose to train the top 2 inception blocks, 
##                 # i.e. we will freeze the first layers and unfreeze the rest:
##                 for layer in model.layers[:len(model.layers)-layer2Freeze]:
##                     layer.trainable = False
##                 for layer in model.layers[len(model.layers)-layer2Freeze:]:
##                     layer.trainable = True
#            
##                 # we need to recompile the model for these modifications to take effect
##                 # we use SGD with a low learning rate
##                 model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
#
##                 # we train our model again (this time fine-tuning the top 2 inception blocks
##                 # alongside the top Dense layers
##                 datagen = ImageDataGenerator(
##                     featurewise_center=True,
##                     featurewise_std_normalization=True,
##                     rotation_range=40,
##                     width_shift_range=0.3,
##                     height_shift_range=0.3,
##                     horizontal_flip=True,
##                     vertical_flip=True)
#
##                 # compute quantities required for featurewise normalization
##                 # (std, mean, and principal components if ZCA whitening is applied)
##                 datagen.fit(x_train)
#
##                 # fits the model on batches with real-time data augmentation:
##                 history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
##                                     steps_per_epoch=len(x_train) / batch_size,
##                                     epochs=epochs_CNN,
##                                     validation_data=(x_test, y_test),
##                                     class_weight = class_weight)
#
##                 ###############################################################
##                 print('\nModel Evaluation\n')
##                 file.write('\nModel Evaluation\n')
##                 y_hat = model.predict(x_test, batch_size=batch_size,verbose=1)
##                 y_pred = np.argmax(y_hat,axis=1)
##                 y_true = np.argmax(y_test,axis=1)
##                 print(classification_report(y_true, y_pred))
##                 file.write(classification_report(y_true, y_pred))
##                 file.write('\n==========================\n')
##                 ###############################################################
        y_pred_Total.extend(y_pred)
        y_true_Total.extend(y_true)
        ###############################################################
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_path+output_name+'_model_accuracy.png')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_path+output_name+'_model_loss.png')
        plt.show()

    #############################################################################    
    print('\n\n ==== FINAL Model Evaluation ==== ')
    file.write('\n\n\n ==== FINAL Model Evaluation ==== ')

    print(classification_report(np.array(y_true_Total), np.array(y_pred_Total)))
    file.write(classification_report(np.array(y_true_Total), np.array(y_pred_Total)))
    file.write('==========================')


    #############################################################################
    class_names = ['Non-malignant','Malignant']
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true_Total, y_pred_Total)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig(log_path+output_name+'_Confusion_matrix.png')


    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(log_path+output_name+'_Normalized_confusion_matrix.png')

    plt.show()

    file.close() 
