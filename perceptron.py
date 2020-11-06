import tqdm
import numpy as np
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_selection import SelectKBest,SelectFromModel,f_classif
from sklearn.metrics import mean_squared_error
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

train_labels = train_features.pop('Seasons')
test_labels = test_features.pop('Seasons')
#data preprocessing
#Standardize features by removing the mean 
#and scaling to unit variance
scaler = preprocessing.StandardScaler().fit(train_features)
train_scaled = scaler.transform(train_features)
test_scaled = scaler.transform(test_features)
#select k best features using SelectKBest rountine
#based on ANOVA F-value between label/feature
train_new = SelectKBest(f_classif, k=5).fit_transform(train_scaled, train_labels)
test_new = SelectKBest(f_classif, k=5).fit_transform(test_scaled, test_labels)

print(train_new)
def perceptron():
    tot_epoch = 50
    learning_rate=[0.01,0.1,1]
    num = 1
    eps = np.arange(1,(tot_epoch//num)+1)*num
    test_sc = np.zeros(tot_epoch//num)
    train_sc = np.zeros(tot_epoch//num)
    test_err = np.zeros(tot_epoch//num)
    train_err = np.zeros(tot_epoch//num)
    cl = ['firebrick','darkgreen','darkblue']
    plt.figure(figsize=(7,5))  
    for k in range(len(learning_rate)):
        for i in tqdm.tqdm(range(tot_epoch//num)):
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            perceptron = Perceptron(max_iter=(i+1)*num,eta0=learning_rate[k])
            clf = OneVsOneClassifier(perceptron)
            clf.fit(train_new, train_labels)
            pred_train = clf.predict(train_new)
            pred_test = clf.predict(test_new)
            train_err[i] = mean_squared_error(train_labels, pred_train)
            test_err[i] = mean_squared_error(test_labels, pred_test)
            train_sc[i]  = clf.score(train_new,train_labels)
            test_sc[i] = clf.score(test_new,test_labels)
            
    
        plt.plot(eps,train_sc,lw=2,label=r'$\eta_0=$'+str(learning_rate[k])+',train R2score',color=cl[k])
        plt.plot(eps,test_sc,lw=2,label=r'$\eta_0=$'+str(learning_rate[k])+',test R2score',color=cl[k],alpha=.5)
    plt.xlabel(r'$Epochs$')
    plt.ylabel(r'$R2 \ Score$')
    plt.legend()
    plt.savefig('./plots/perceptron_one_vs_one.eps')
    plt.show()

perceptron()