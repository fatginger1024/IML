
import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show,savefig


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

features = data.drop('Rented Bike Count',1)
labels = data['Rented Bike Count']

# define search space
kwargs = {}
kwargs['criterion'] = ['mse', 'mae']
kwargs['max_depth'] = [2,4,8,16]
kwargs['min_samples_leaf'] = [1,2,3,4]
kwargs['min_weight_fraction_leaf'] = [0.,.1,.2,.3,.4]
kwargs['max_features'] = ['auto','sqrt','log2']
kwargs['min_samples_split'] = [2,4,6]


def forest():
    fig  = figure(figsize=(10,6))
    for m,key in enumerate(kwargs.keys()):
        ax1 = fig.add_subplot(2,3,m+1)
        
        args = kwargs[key]
        R2_Score = np.zeros(len(args))
        oob_Score = np.zeros(len(args))

        for i,arg in enumerate(args):
            print("m=",m,"i=",i)
            print("="*80)
            regr = RandomForestRegressor(**{key:arg},oob_score=True,n_estimators=100)
            regr.fit(features, labels)
            oob_score = regr.oob_score_
            scores = cross_validate(regr, features, labels, cv=5,
                                 scoring=('r2'),
                                 return_train_score=True)

            R2_Score[i] = scores['test_score'].mean()
            oob_Score[i] = oob_score
            
        
    
        ax1.plot(args,R2_Score,lw=2,label='R2 Score')
        ax1.plot(args,oob_Score,lw=2,label='oob Score')
        ax1.set_xlabel(key)
        ax1.set_ylabel('Score')    
        ax1.legend()
    plt.tight_layout()
    savefig('./plots/forest.eps')
    show()



    
forest()