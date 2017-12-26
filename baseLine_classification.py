from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils import np_utils

import numpy as np
import glob
import random
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

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
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#     print(cm)

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

def calc_scores(clf, X, y,clf_name,class_names=['0','1'], cv=10, plot_cnf_matrix=True, figsize=(5,5), normalize=True):
#     ## Compute Evaluation Metrics
#     scores = cross_val_score(clf, X, y, cv=cv)
#     print(clf_name,np.mean(scores))
        
    scoring = ['accuracy', 'precision','recall']
    scores = cross_validate(clf, X, y, cv=cv ,scoring=scoring)
    print('\n',clf_name,'\ntest_accuracy',np.array(np.mean(scores['test_accuracy'])),
                        '\ntest_precision',np.array(np.mean(scores['test_precision'])),                     
                        '\ntest_recall',np.array(np.mean(scores['test_recall'])))
    
    
    ## Compute Confusion matrix
    y_pred = cross_val_predict(clf,X,y,cv=cv)
    cnf_matrix = confusion_matrix(y,y_pred)
    
    # Plot Confusion matrix
    if (plot_cnf_matrix):
        plt.figure(figsize=figsize)
        if (normalize):
            title = ' Normalized confusion matrix'
        else:
            title = ' non-Normalized confusion matrix'
        class_names = ['0']
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=normalize,title=clf_name+title)
        plt.show()
    
def load_Ds(folder_Path):
    ds = np.load(folder_Path)
    X = np.array(ds[:,:-1])
    y = np.array(list(map(int,ds[:,-1])))
    return X,y

def shuffle(x_data, y_data):
    c = list(zip(x_data, y_data))
    random.shuffle(c)
    X,y = zip(*c)
    X = np.array(X)
    y = np.array(y)
    return X,y

def simplifly(X,y):   
    idx_list = [idx for idx, k in enumerate(y) if k!=2]    
    X = np.array(X[idx_list])
    y = np.array(y[idx_list])
    
    # Binary problem
    y = np.array([0 if k<2 else k for k in y])
#     y = np.array([1 if k==2 else k for k in y])        
    y = np.array([1 if k>2 else k for k in y])
    
    # Multiclas problem
#     y = np.array([2 if k==3 else k for k in y])
#     y = np.array([3 if k==4 else k for k in y])
    return X, y

###############################################################################

#Especifique se a matriz de confusão deve ser plotada
plot_cnf_matrix= True
#Especifique se a matriz de confusão plotada deve ser normalizada
normalize =True 
#Especifique o tamanho matriz de confução deve ser plotada
figsize = (5,5) # <<normal # figsize = (10,10) <<<Grande
# KFOLD 
cv = 5

# Especifique os classificadores a serem utilizados
clf_list = [ NearestCentroid(),
             RandomForestClassifier(n_estimators=20,min_samples_split=2, max_depth=3, class_weight={0: 1 ,1: 1}),
             MultinomialNB(),
             GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
             svm.SVC(kernel='linear', C=1)]

clf_name_list = [ 'NearestCentroid',
                  'RandomForest Classifier',
                  'Multinomial NB',
                  'Gradient Boosting Classifier',
                  'SVM']

# SO RANDOM FOREST
clf_list = clf_list[1:2]
clf_name_list = clf_name_list[1:2]

# MENOS SVM
# clf_list = clf_list[:-1]
# clf_name_list = clf_name_list[:-1]


#Especifique a topologia a ser utilizada.
model_name_list = ['VGG16','VGG19','MobileNet','ResNet50','InceptionV3','Xception']
model_name_list = ['ResNet50']

for model_name in model_name_list:
    print("###################################################################")
    print(model_name)
    print("###################################################################")
   
    ## Especifique o caminho da pasta onde estão as imagens a serem processadas.
    src_path = './Out_Folder/LIDC-IDRI/16x16x16/3Axis/max/'+model_name+'.npy'
#     src_path = './Out_Folder/LIDC-IDRI/32x32x32_2/3Axis/max/'+model_name+'.npy'
#     src_path = './Out_Folder/LIDC-IDRI/64x64x64/3Axis/max/'+model_name+'.npy'
    
    X, y  = load_Ds(src_path)
    print("Loaded DS has:",X.shape,"intances",y.shape,"labels")

    X, y  = simplifly(X,y)
    X, y  = shuffle(X, y)
    print("Simplifly DS has:",X.shape,"intances",y.shape,"labels")

    num_classes = len(np.unique(y))
    classes = [np.sum([1 for x in y if x==k]) for k in range(num_classes)]
    print("Number of Classes:",num_classes,"with",classes,"elements respectively.")class_names
    
    

    for clf,clf_name in zip(clf_list,clf_name_list):    
        calc_scores(clf, X, y,clf_name, cv, plot_cnf_matrix, figsize, normalize)