import tqdm
import numpy as np
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/tardis/Downloads/bikes_new.csv')
#split dataset into training set/test set
train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)

#split features and labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('labels')
test_labels = test_features.pop('labels')
#data preprocessing
#Standardize features by removing the mean 
#and scaling to unit variance
scaler = preprocessing.StandardScaler().fit(train_features)
train_scaled = scaler.transform(train_features)
test_scaled = scaler.transform(test_features)

#perform one vs rest multiclassification
def one_vs_rest():
    tot_epoch = 5000
    num = 100
    eps = np.arange(1,(tot_epoch//num)+1)*num
    test_sc = np.zeros(tot_epoch//num)
    train_sc = np.zeros(tot_epoch//num)
    test_err = np.zeros(tot_epoch//num)
    train_err = np.zeros(tot_epoch//num)

    for i in tqdm.tqdm(range(tot_epoch//num)):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        params_ = {"estimator__max_iter":(i+1)*num}
        clf = OneVsRestClassifier(SVC(gamma='auto'))
        clf.set_params(**params_)
        clf.fit(train_scaled, train_labels)
        pred_train = clf.predict(train_scaled)
        pred_test = clf.predict(test_scaled)
        train_err[i] = mean_squared_error(train_labels, pred_train)
        test_err[i] = mean_squared_error(test_labels, pred_test)
        train_sc[i]  = clf.score(train_scaled,train_labels)
        test_sc[i] = clf.score(test_scaled,test_labels)
    plt.figure(figsize=(7,5))  
    plt.plot(eps,train_sc,lw=2,label='train R2score')
    plt.plot(eps,test_sc,lw=2,label='test R2score')
    plt.xlabel(r'$Epochs$')
    plt.ylabel(r'$R2 \ Score$')
    plt.legend()
    plt.savefig('./plots/multiclass_one_vs_rest.eps')
    plt.show()
    
    plt.figure(figsize=(7,5))  
    plt.plot(eps,train_err,lw=2,label='train error')
    plt.plot(eps,test_err,lw=2,label='test error')
    plt.xlabel(r'$Epochs$')
    plt.ylabel(r'$MSE$')
    plt.legend()
    plt.savefig('./plots/multiclass_one_vs_rest_err.eps')
    plt.show()
    
def one_vs_one():
    
    tot_epoch = 5000
    num = 100
    eps = np.arange(1,(tot_epoch//num)+1)*num
    test_sc = np.zeros(tot_epoch//num)
    train_sc = np.zeros(tot_epoch//num)
    test_err = np.zeros(tot_epoch//num)
    train_err = np.zeros(tot_epoch//num)

    for i in tqdm.tqdm(range(tot_epoch//num)):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        params_ = {"estimator__max_iter":(i+1)*num}
        clf = OneVsOneClassifier(LinearSVC(random_state=0))
        clf.set_params(**params_)
        clf.fit(train_scaled, train_labels)
        pred_train = clf.predict(train_scaled)
        pred_test = clf.predict(test_scaled)
        train_err[i] = mean_squared_error(train_labels, pred_train)
        test_err[i] = mean_squared_error(test_labels, pred_test)
        train_sc[i]  = clf.score(train_scaled,train_labels)
        test_sc[i] = clf.score(test_scaled,test_labels)
    plt.figure(figsize=(7,5))  
    plt.plot(eps,train_sc,lw=2,label='train R2score')
    plt.plot(eps,test_sc,lw=2,label='test R2score')
    plt.xlabel(r'$Epochs$')
    plt.ylabel(r'$R2 \ Score$')
    plt.legend()
    plt.savefig('./plots/multiclass_one_vs_one.eps')
    plt.show()
    
    plt.figure(figsize=(7,5))  
    plt.plot(eps,train_err,lw=2,label='train error')
    plt.plot(eps,test_err,lw=2,label='test error')
    plt.xlabel(r'$Epochs$')
    plt.ylabel(r'$MSE$')
    plt.legend()
    plt.savefig('./plots/multiclass_one_vs_one_err.eps')
    plt.show()
    
one_vs_rest()   
one_vs_one()