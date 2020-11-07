import numpy as np
import pandas as pd

from sklearn import cluster
from sklearn import preprocessing
from sklearn.decomposition import PCA


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
scaler = preprocessing.StandardScaler().fit(train_features)
train_scaled = scaler.transform(train_features)
test_scaled = scaler.transform(test_features)

def cluster_PCA():
    agglo = cluster.FeatureAgglomeration(n_clusters=2)
    cluster_out = agglo.fit_transform(train_scaled)
    pca = PCA(n_components=2)
    out = pca.fit_transform(cluster_out)
    print("="*70)
    print("="*30,"Method 1","="*30)
    print("="*70)
    print("*"*70)
    print("Clustering")
    print("*"*70)
    print("Number of clusters:\n ",agglo.n_clusters_)
    print("Cluster labels:\n ",agglo.labels_)
    print("Number of leaves: ",agglo.n_leaves_)
    print("Number of connected components: ",agglo.n_connected_components_)
    print("Children of each node: ",agglo.children_)
    print("*"*70)
    print("PCA")
    print("*"*70)
    print("Components of principal axes:\n",pca.components_)
    print("Amount of variance:\n ",pca.explained_variance_)
    print("Percentage of variance:\n ",pca.explained_variance_ratio_)
    print("Singular values:\n ",pca.singular_values_)
    print("Per-feature empirical mean:\n ",pca.mean_)
    print("*"*70)
    
    print("Transformed data with cluster then PCA: \n",out)
    print("="*70)
    
def PCA_cluster():
    pca = PCA(n_components=2)
    pca_out = pca.fit_transform(train_scaled)
    agglo = cluster.FeatureAgglomeration(n_clusters=2)
    out = agglo.fit_transform(pca_out)
    print("="*70)
    print("="*30,"Method 2","="*30)
    print("="*70)
    print("*"*70)
    print("PCA")
    print("*"*70)
    print("Components of principal axes:\n",pca.components_)
    print("Amount of variance:\n ",pca.explained_variance_)
    print("Percentage of variance:\n ",pca.explained_variance_ratio_)
    print("Singular values:\n ",pca.singular_values_)
    print("Per-feature empirical mean:\n ",pca.mean_)
    print("*"*70)
    print("Clustering")
    print("*"*70)
    print("Number of clusters:\n ",agglo.n_clusters_)
    print("Cluster labels:\n ",agglo.labels_)
    print("Number of leaves: ",agglo.n_leaves_)
    print("Number of connected components: ",agglo.n_connected_components_)
    print("Children of each node: ",agglo.children_)
    print("*"*70)
    
    print("Transformed data with PCA then cluster: \n",out)
    print("="*70)
    
    
    
cluster_PCA()
PCA_cluster()
    
    
    