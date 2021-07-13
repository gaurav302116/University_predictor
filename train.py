import numpy as np
import pandas as pd
import random
import csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report,confusion_matrix,accuracy_score
import pickle 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def import_data():
    X=np.genfromtxt('train_X_p1.csv',delimiter=',',dtype=np.float64,skip_header=1)
    Y=np.genfromtxt('train_Y_p1.csv',delimiter=',',dtype=np.float64,skip_header=1)
    return X,Y

X,Y=import_data()
Y=np.insert(Y,0,1)


# the scaler object (model)
scaler = preprocessing.StandardScaler()# fit and transform the data
scaled_data = scaler.fit_transform(X) 
X = scaler.transform(X)
"""
Y1=Y1.reshape(len(Y1),1)
#Y1=np.insert(Y1,0,1)
X=np.concatenate((X1,Y1),axis=1)
np.random.shuffle(X)
#print(X)
Y=X[:,11]
X=X[:,[1,3,4,6,7,8,9,10]]
#Y=X[:,11]
"""
X=X[:,[1,3,4,6,7,8,9,10]]
X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size = 0.01
                                                   ,random_state = 2018)
#model fitting
svclassifier=SVC(C=5,kernel='rbf',gamma=0.2)
svclassifier.fit(X_train,y_train)
    
#y_pred1=svclassifier.predict(X_train)
    
y_pred=svclassifier.predict(X_test)
score=f1_score(y_test,y_pred)
#print(score)

q=accuracy_score(y_test,y_pred)


matrix=confusion_matrix(y_test,y_pred)



# Save the trained model as a pickle string.  
pkl_file='MODEL_FILE.sav'
with open(pkl_file,'wb') as file:
    pickle.dump(svclassifier,file) 





