from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm

import numpy as np
import glob
import random
import matplotlib.pyplot as plt

def load_Ds(folder_Path):
    X = []; y= []
    for malignancyID in range(1,6):
        for file_path in glob.glob(folder_Path+'/'+str(malignancyID)+'/*'):
            X.append(np.load(file_path)[8,:,:].flatten())
            if(malignancyID <3):
                y.append(0)
            if(malignancyID >3):
                y.append(1)
    return X,y

def shuffle(x_data, y_data):
    c = list(zip(x_data, y_data))
    random.shuffle(c)
    return zip(*c)


folder_Path = '/home/raul/PROJECTS/LIDC-IDRI/Out_Folder_PYTHON'
X, y  = load_Ds(folder_Path)
X, y  = shuffle(X, y)
X = np.array(X)
y = np.array(y)

print(X.shape)
print(len(y),"nodules were loaded")

clf = NearestCentroid()
scores = cross_val_score(clf, X, y, cv=2 )
print('NearestCentroid',np.mean(scores))

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=2)
print('SVM',np.mean(scores))