# ann.ipbnf
import numpy as np
from sklearn.datasets import load_breast_cancer
#loading breast cancer dataset from scikit learn library for classification into two classes!
from sklearn.linear_model import Perceptron
#loading the perceptron model redily availble in sklearn library for classification
cancerdataset=load_breast_cancer()
print(cancerdataset)
print(cancerdataset.DESCR) #DESCR MEANS DISCRITIVE STACISTICS FORMAT
X=cancerdataset.data
y=cancerdataset.target
print(y)  #perceptron is a ann model used to classify the dataset into two classess 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.6,stratify=y,random_state=77)
classifier=Perceptron()
classifier.fit(X_train,y_train)
score=classifier.score(X_test,y_test)
print(score)
