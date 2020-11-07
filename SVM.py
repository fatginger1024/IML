import tqdm
import numpy as np
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
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
scaler = preprocessing.StandardScaler().fit(train_features)
train_scaled = scaler.transform(train_features)
test_scaled = scaler.transform(test_features)
#select k best features using SelectKBest rountine
#based on ANOVA F-value between label/feature
train_scaled = SelectKBest(f_classif, k=5).fit_transform(train_scaled, train_labels)
test_scaled = SelectKBest(f_classif, k=5).fit_transform(test_scaled, test_labels)



def svm():
    tot_epoch = 10000
    kernel=['poly', 'rbf', 'sigmoid']
    num = 1000
    eps = np.arange(1,(tot_epoch//num)+1)*num
    test_sc = np.zeros(tot_epoch//num)
    train_sc = np.zeros(tot_epoch//num)
    test_err = np.zeros(tot_epoch//num)
    train_err = np.zeros(tot_epoch//num)
    cl = ['firebrick','indigo','darkgreen']
    plt.figure(figsize=(7,5))  
    for k in range(len(kernel)):
        for i in tqdm.tqdm(range(tot_epoch//num)):
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            svr = SVR(kernel=kernel[k], gamma=0.1,max_iter=(i+1)*num)
            svr.fit(train_scaled, train_labels)
            pred_train = svr.predict(train_scaled)
            pred_test = svr.predict(test_scaled)
            train_err[i] = mean_squared_error(train_labels, pred_train)
            test_err[i] = mean_squared_error(test_labels, pred_test)
            train_sc[i]  = svr.score(train_scaled,train_labels)
            test_sc[i] = svr.score(test_scaled,test_labels)
            
    
        plt.plot(eps,train_sc,lw=2,label=r'$kernel=$'+str(kernel[k])+',train R2score',color=cl[k])
        plt.plot(eps,test_sc,lw=2,label=r'$kernel=$'+str(kernel[k])+',test R2score',color=cl[k],alpha=.5)
    plt.xlabel(r'$Epochs$')
    plt.ylabel(r'$R2 \ Score$')
    plt.legend()
    plt.savefig('./plots/svm_one_vs_one.eps')
    plt.show()
    
    
def cv_score():
    kernel=['poly', 'rbf', 'sigmoid']
    sc = np.zeros(len(kernel))
    plt.figure(figsize=(7,5))  
    for k in range(len(kernel)):
        svr = SVR(kernel=kernel[k],gamma='auto')
        #since cross validation algorithm does train-test-validation splitting
        #we do not have to prer split the data set 
        X = np.vstack((train_scaled,test_scaled))
        y = np.hstack((train_labels,test_labels))
        svr.fit(X, y)
        sc[k] = np.mean(cross_val_score(svr, X,y,cv=5))
    plt.plot(kernel,sc,lw=2,color='orange')
    plt.xlabel(r'$Kernel$')
    plt.ylabel(r'$CV \ Score$')
    plt.legend()
    plt.savefig('./plots/svm_cv_score.eps')
    plt.show()
    

svm()
cv_score()



