#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:32:22 2017

@author: raul
"""
import os
import numpy as np
import random
import scipy
import cv2
import tensorflow as tf
    
from sklearn.metrics import confusion_matrix
import itertools

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalMaxPooling2D , Flatten ,Input, Dropout
from keras import backend as K
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import optimizers


from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
            
class LIDC_IDRI:
    
    
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

#    def load_Img(filepath):
#        #LOAD image to be deep extracted         
#        if (filepath.endswith('jpg')):
#            img = image.load_img(filepath)
#        else:
#            if (filepath.endswith('txt')):
#                img = np.loadtxt(filepath).copy()
#            if (filepath.endswith('npy')):
#                img = np.load(filepath).copy()
#        return img

    def simplify_Prob(x_data,y_data):
        new_y = [];new_x = [];
        for x,y in zip(x_data,y_data):
            if (y>2):
                new_y.append(1)
                new_x.append(x)
            if (y<2):
                new_y.append(0)
                new_x.append(x)
        return new_x,new_y,2
    
    def load_data(folderPath,target_size,axis_name,mid,multiclass=False,force_balance=False,shuffle=True,clone=False):
        print("folderPath",folderPath)
        x = []; y = []
        for idx,classID in enumerate(os.listdir(folderPath)):
            pathClassID = folderPath+'/'+classID        
            #For each nodule file:
            for subdir, dirs, files in os.walk(pathClassID):
                for name in files:
                    filePath = subdir + os.sep + name
                    img = LIDC_IDRI.load_img(filePath,target_size,axis_name,mid)
                    
                    x.append(img)
                    y.append(int(classID))
                    
        nClass = len(os.listdir(folderPath))
    
        if(multiclass == False):
            x,y,nClass = LIDC_IDRI.simplify_Prob(x,y)
        if(force_balance == True):
            x,y = LIDC_IDRI.force_balance(x, y,nClass)
        if(shuffle == True):
            print(len(x))
            x,y = LIDC_IDRI.shuffle(x, y)
        if(clone == True):
            x,y = LIDC_IDRI.clone_class(x,y,1,2)
        
        return np.array(x),np.array(y),nClass
    
    def force_balance(x_data, y_data,num_classes):    
        sizeBalancedDS = np.min([np.sum([1 for i in y_data if i == x]) for x in range(num_classes)])
        temp_x = [];temp_y = []; count = np.zeros(num_classes)
        for xd ,yd in zip(x_data, y_data):
            for i in range(num_classes):
                if (count[i] < sizeBalancedDS and yd == i):
                    temp_x.append(xd)
                    temp_y.append(i)
                    count[i]+=1
        return temp_x ,temp_y      
    
    def clone_class(x_data, y_data,class_to_clone,times):        
        y_clone = np.array([i for i in y_data if i == class_to_clone])
        x_clone = np.array([x_data[idx] for idx, i in enumerate(y_data) if i == class_to_clone])
            
        print("original size ",len(x_data),len(x_clone))
        
        temp_x = []; temp_y = [];
        temp_x.extend(x_data)
        temp_y.extend(y_data)
        
        for k in range(times): 
            temp_x.extend(x_clone)
            temp_y.extend(y_clone)
            
        print("new size ",[np.sum([1 for i in temp_y if i == x]) for x in [0,1]])

        return temp_x ,temp_y     

    
    def shuffle(x_data, y_data):
        c = list(zip(x_data, y_data))
        random.shuffle(c)
        return zip(*c)

def edit_model(base_model,num_classes):
    # Add a Flatten layer
    x = base_model.output
    x = Flatten()(x)

#    # Add a fully-connected layer
#    x = Dense(512, activation='relu')(x)
#    x = Dropout(0.8)(x)
#    x = Dense(128, activation='relu')(x)

    # Add a logistic layer (to match the number of classes)
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
 #              print('name ',len(model.layers[]))
    
     # first: train only the top layers (which were randomly initialized)
     # i.e. freeze all convolutional base_model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    return model
def buildModel(modelName,input_tensor,num_classes):
    
    if ('VGG16' is modelName):
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)    
    if ('VGG19' is modelName):
        base_model = VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
    if ('MobileNet' is modelName):
        base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=input_tensor)  
    if ('ResNet50' is modelName):
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)       
    if ('InceptionV3' is modelName):
        base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
    if ('Xception' is modelName):
        base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)  
    if('InceptionResNetV2' is modelName):
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
        
    # Adapt model to new output(number of classes)
    return edit_model(base_model,num_classes)
     
    
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
    
def init_log_var(plane,modelName,target_size,layer2Freeze):
    """ this function initialize the log variables and paths""" 
    output_name = str(np.random.randint(10000000))+'_'+plane+'_'+modelName+'_'+str(target_size)+'_'+str(layer2Freeze)
    print(output_name)
    
    log_path = output_path+'_'+output_name+'/'
    os.makedirs(log_path, exist_ok=True)
    
    file = open(log_path+output_name+".txt","w") 
    file.write(output_name+'\n')
    return file,log_path,output_name

def log_temp_metrics(model,log_name,x_test,y_test,file):
    # METRICS            
    print('\nModel Evaluation '+log_name+'\n')
    file.write('\nModel Evaluation '+log_name+'\n')
    y_hat = np.array(model.predict(x_test))
    
    if (len(y_test.shape) > 1):
        y_pred = np.argmax(y_hat,axis=1)
        y_true = np.argmax(y_test,axis=1)
    else:
        y_pred = y_hat
        y_true = y_test
        
    print(classification_report(y_true, y_pred))
    file.write(classification_report(y_true, y_pred))
    file.write('\n==========================\n')   
    return y_true,y_pred

def log_final_metrics(log_name,y_pred,y_true,file):
    # METRICS            
    print('\n\n ==== FINAL Model Evaluation '+log_name+' ====\n ')
    file.write('\n\n ==== FINAL Model Evaluation '+log_name+' ====\n ')
    print(classification_report(np.array(y_true), np.array(y_pred)))
    file.write(classification_report(np.array(y_true), np.array(y_pred)))
    file.write('==========================')
    return True

def log_cofusion_matrix(y_true_Total, y_pred_Total,file_path, file_name):
    os.makedirs(file_path, exist_ok=True)
    file_path =file_path+file_name
    
    class_names = ['Non-malignant','Malignant']
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true_Total, y_pred_Total)
    np.set_printoptions(precision=3)
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
    plt.savefig(file_path+'_Confusion_matrix.png')
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plt.savefig(file_path+'_Normalized_confusion_matrix.png')
    
    plt.show()

def plotHistory(history,file_path,file_name):
    os.makedirs(file_path, exist_ok=True)
    file_path =file_path+file_name
    
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(file_path+'_model_accuracy.png')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(file_path+'_model_loss.png')
    plt.show()
    return True

def freezeModel(model,layer2Freeze):
    # we chose to train the top 2 inception blocks, 
    # i.e. we will freeze the first layers and unfreeze the rest:
    for layer in model.layers[:len(model.layers)-layer2Freeze]:
        layer.trainable = False
    for layer in model.layers[len(model.layers)-layer2Freeze:]:
        layer.trainable = True
    return model


def train_model(model,x_train,y_train,x_test,y_test,batch_size,epochs,class_weight,opt=None,datagen=None):
    # configure otimizer
    if (opt == None):
        opt=optimizers.Adam(lr=1e-03, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=opt, loss='mean_squared_error',metrics=['accuracy'])
    
    # train the model on the new data for a few epochs
    if (datagen == None):
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=40,
            width_shift_range=0.6,
            height_shift_range=0.6,
            horizontal_flip=True,
            vertical_flip=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    # fits the model on batches with real-time data augmentation:
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                        verbose=1, validation_data=(x_test, y_test),
                        class_weight = class_weight)
    return history


def save_model(model,out_path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(out_path+"model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(out_path+"model.h5")
    print("Saved model to disk")
    return True

def extract_features(model,x_test):
    layerName = model.layers[-2].name                          
    x = model.get_layer(layerName).output
    new_model = Model(inputs=model.input, outputs=x)
    return new_model.predict(x_test)

###############################################################################
###############################################################################
###############################################################################

vol_size = 16

srcPath_list = ['/home/raul/PROJECTS/LIDC-IDRI/Out_Folder_NODULES_NPY/Median/'+str(vol_size)+'x'+str(vol_size)+'x'+str(vol_size)]
class_weight = {0: 1 ,1: 4}
#class_weight = None

# Loading Variables
multiclass = False
force_balance = False
shuffle = True
clone = False

clf_name_list = ['Nearest_Centroid',
                 '1 Nearest_Neighbors',
                 '3 Nearest_Neighbors ',
                 '5 Nearest_Neighbors ',
                 'Random_Forest_Classifier',
                 'Multinomial_NB',
                 'Gradient_Boosting_Classifier',
                 'SVM linear',
                 'SVM poly',
                 'SVM rbf',
                 'SVM sigmoid']
    
clf_list = [ NearestCentroid(), 
             KNeighborsClassifier(n_neighbors=1),
             KNeighborsClassifier(n_neighbors=3), 
             KNeighborsClassifier(n_neighbors=5),
             RandomForestClassifier(n_estimators=20, max_depth=6, random_state=0,class_weight=class_weight),
             MultinomialNB(),
             GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
             svm.SVC(kernel='linear',class_weight=class_weight),
             svm.SVC(kernel='poly',class_weight=class_weight),
             svm.SVC(kernel='rbf',class_weight=class_weight),
             svm.SVC(kernel='sigmoid',class_weight=class_weight)]

#which plane
plane_list = ['Axial']

# Resize source image
target_size_list = [64]

# Base model to be fine-tuned 
modelName_list = ['VGG16']

# fine-tuning from layer
layer2Freeze_list = [3]

# number of epochs 
epochs_FC = 10
epochs_CNN = 50

# number of samples in a batch
batch_size = 32

# number of folds
K_fold = 5


out_path = './Framework_outputs/'
fix_stages_names_list = ['Fully_Connected','ConvNet']
#fix_stages_names_list = ['ConvNet']
###############################################################################

for srcFolder in srcPath_list:
    output_path= out_path+srcFolder.split('/')[-1]
    for plane in plane_list:        
        for modelName,target_size,layer2Freeze in zip(modelName_list,target_size_list,layer2Freeze_list):
            ###################################################################
            ## Init log variables
            file, log_path,output_name = init_log_var(plane,modelName,target_size,layer2Freeze)
            ###################################################################
            #Load Dataset    
            x_data,y_data,num_classes = LIDC_IDRI.load_data(srcFolder,target_size,plane,int(vol_size/2),multiclass,force_balance,shuffle,clone)

            print('Loaded DS has:', len(x_data), 'samples /',num_classes,'classes')
            print([np.sum([1 for i in y_data if i == x]) for x in range(num_classes)])
            file.write('Loaded DS has: '+str(len(x_data))+' samples | '+str(num_classes)+' classes\n') 
            ###################################################################
            ## Onehot labels
            y_data_hot = np_utils.to_categorical(y_data, num_classes)
            ###################################################################
            ## Log Variables
            y_pred_list = []
            y_true_list = []
            
            num_clf = len(clf_name_list)
            len_fix_stages = len(fix_stages_names_list)
            for idx in range(num_clf*2+len_fix_stages):
                y_pred_list.append([])
                y_true_list.append([])
                
            ###################################################################
            skf = StratifiedKFold(n_splits=K_fold)
            for fold , (train, test) in enumerate(skf.split(x_data, y_data)):
                #############################
                x_train = x_data[train]
                y_train = y_data_hot[train]
                x_test  = x_data[test]
                y_test  = y_data_hot[test]
                #############################
                print('\n###########################################################################################')
                print('K-fold number: ',fold+1,'\n')
                file.write('\n\nK-fold number: '+str(fold+1)+'\n')
                input_tensor = Input(shape=(x_train[0].shape))
                print(x_train[0].shape)
                
                strFoldID= str(fold+1)+"fold"
                file_path_fold = log_path+'/'+strFoldID+'/'
                idx = 0
                ###############################################################
                # Create the base pre-trained model and adapt model to new output(number of classes)
                model = buildModel(modelName,input_tensor,num_classes)   
                ###############################################################
                
                ## Classifiers in the baseline model
                ###############################################################
                stage_name ='TRANSFER_LEARNING'
                x_TF_train = extract_features(model,x_train)
                x_TF_test = extract_features(model,x_test)
                ###############################################################
                for clf ,clf_name in zip(clf_list,clf_name_list):            
                    file_name = stage_name+'_'+clf_name
                    ###########################################################
                    clf.fit(np.array(x_TF_train),np.array(y_data[train]))
                    ###########################################################
                    # Compute metrics of process of fine tune
                    y_true, y_pred = log_temp_metrics(clf,file_name,x_TF_test,np.array(y_data[test]),file)
                    log_cofusion_matrix(y_true, y_pred,file_path_fold, file_name)
                    ###########################################################
                    y_pred_list[idx].extend(y_pred)
                    y_true_list[idx].extend(y_true)
                    idx+=1
                    ###########################################################    
                x_TF_train = []
                x_TF_test = []
                ###############################################################

                
                # Fully Connected 
                ###############################################################
                stage_name="FC"
                ###############################################################
                # Train the last last layer of the model (Fully connected)
                history = train_model(model,x_train,y_train,x_test,y_test,batch_size,epochs_FC,class_weight)
                plotHistory(history,file_path_fold,stage_name)
                if('Fully_Connected' in fix_stages_names_list):
                    # Compute the train metrics  
                    y_true, y_pred = log_temp_metrics(model,stage_name,x_test,y_test,file)   
                    log_cofusion_matrix(y_true, y_pred, file_path_fold, stage_name)
                    ###############################################################
                    y_pred_list[idx].extend(y_pred)
                    y_true_list[idx].extend(y_true)
                    idx+=1
                    ###############################################################
            
            
                ## CNN
                ###############################################################
                stage_name="CNN"
                ###############################################################
                model = freezeModel(model,layer2Freeze)
                # Fine tune some layers of the model
                history = train_model(model,x_train,y_train,x_test,y_test,batch_size,epochs_CNN,class_weight)
                plotHistory(history,file_path_fold,stage_name)
                # Compute metrics of the process of fine tune
                y_true, y_pred = log_temp_metrics(model,stage_name,x_test,y_test,file)
                log_cofusion_matrix(y_true, y_pred, file_path_fold,stage_name)
                ###############################################################
                y_pred_list[idx].extend(y_pred)
                y_true_list[idx].extend(y_true)
                idx+=1
                ###############################################################    
                
                ## Classifiers
                ###############################################################
                stage_name ='FINETUNE'
                ###############################################################
                x_FT_train = extract_features(model,x_train)
                x_FT_test = extract_features(model,x_test)
                ###############################################################
                for clf ,clf_name in zip(clf_list,clf_name_list):  
                    file_name = stage_name+'_'+clf_name
                    ###########################################################
                    clf.fit(np.array(x_FT_train),np.array(y_data[train]))
                    ###########################################################
                    # Compute metrics of process of fine tune
                    y_true, y_pred = log_temp_metrics(clf,file_name,x_FT_test,np.array(y_data[test]),file)
                    log_cofusion_matrix(y_true, y_pred, file_path_fold, file_name)
                    ###########################################################
                    y_pred_list[idx].extend(y_pred)
                    y_true_list[idx].extend(y_true)
                    idx+=1
                ###############################################################
                x_FT_train = []
                x_FT_test = []
                ###############################################################
                
            ################################################################### 
            log_path = log_path+'/FINAL/'
            for idx, clf_name in enumerate(clf_name_list):
                log_final_metrics(clf_name,y_pred_list[idx],y_true_list[idx],file)
                log_cofusion_matrix(y_pred_list[idx],y_true_list[idx], log_path,"TRANFER_LEARNING_FINAL_"+clf_name)
            
            for idx, stage_name in enumerate(fix_stages_names_list):
                idx_s = idx+len(clf_name_list)
                log_final_metrics(clf_name,y_pred_list[idx_s],y_true_list[idx_s],file)
                log_cofusion_matrix(y_pred_list[idx_s],y_true_list[idx_s], log_path,"FINAL_"+stage_name)
            
            for idx, clf_name in enumerate(clf_name_list):
                idx_FT = idx +len(clf_name_list)+2
                log_final_metrics(clf_name,y_pred_list[idx_FT],y_true_list[idx_FT],file)
                log_cofusion_matrix(y_pred_list[idx_FT],y_true_list[idx_FT], log_path,"FINETUNE_FINAL_"+clf_name)
            
            file.close()
            ###################################################################
            save_model(model,log_path)
            ###################################################################
            


    