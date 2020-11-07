import tqdm
import numpy as np
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt


data = pd.read_csv('./bikes_seoul.csv')

#Date should be converted to array of floats
def date_to_float(date):
    arr = np.zeros(len(date))
    for i,day in enumerate(date):
        nums = np.asarray(day.split('/')).astype(int)
        num = nums[0]+nums[1]*1e2+nums[2]*1e4
        arr[i] = num
    
    return arr.astype(int)

date = np.array(data['Date'])
date = date_to_float(date) 

#convert categorical columns to floats
data['Seasons'] =  data['Seasons'].map({'Spring':1, 'Summer':2, 'Autumn':3, 'Winter':4})
data['Holiday'] =  data['Holiday'].map({'Holiday':1, 'No Holiday':0})
data['Functioning Day'] = data['Functioning Day'].map({'Yes':1, 'No':0})
data['Date'] = date


train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)

#split features and labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Rented Bike Count')
test_labels = test_features.pop('Rented Bike Count')
#data preprocessing
#Standardize features by removing the mean 
#and scaling to unit variance
scaler = preprocessing.StandardScaler(with_std=False).fit(train_features)
train_scaled = scaler.transform(train_features)
test_scaled = scaler.transform(test_features)
#select k best features using SelectKBest rountine
#based on ANOVA F-value between label/feature
train_new = SelectKBest(f_classif, k=5).fit_transform(train_scaled, train_labels)
test_new = SelectKBest(f_classif, k=5).fit_transform(test_scaled, test_labels)

pca = PCA(n_components=2)
out = pca.fit_transform(train_scaled)
print("Without feature selection ...")
print("Components of principal axes:\n",pca.components_)
print("Amount of variance:\n ",pca.explained_variance_)
print("Percentage of variance:\n ",pca.explained_variance_ratio_)
print("Singular values:\n ",pca.singular_values_)
print("Per-feature empirical mean:\n ",pca.mean_)
print("Transformed data:\n ",out)

pca = PCA(n_components=2)
out = pca.fit_transform(train_new)
print("With feature selection ...")
print("Components of principal axes:\n",pca.components_)
print("Amount of variance:\n ",pca.explained_variance_)
print("Percentage of variance:\n ",pca.explained_variance_ratio_)
print("Singular values:\n ",pca.singular_values_)
print("Per-feature empirical mean:\n ",pca.mean_)
print("Transformed data:\n ",out)


