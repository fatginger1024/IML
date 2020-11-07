import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


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
kwargs['min_weight_fraction_leaf'] = [0.,.1,.2,.3]
kwargs['max_features'] = ['auto','sqrt','log2']
kwargs['min_samples_split'] = [2,4,6]


model = RandomForestRegressor(oob_score=True,n_estimators=100)

# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search
search = RandomizedSearchCV(model, kwargs, n_iter=500, scoring='r2', n_jobs=-1, cv=cv, random_state=1)

# execute search
result = search.fit(features, labels)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)