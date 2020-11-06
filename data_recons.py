import numpy as np
import pandas as pd

#########################################################################################
# Data proprocessing 
#########################################################################################
#reading dataset
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

#########################################################################################
#turn the problem into a classification problem
#########################################################################################
Counts = np.array(data['Rented Bike Count'])
#get bin edges of the 5 bins
hist,bins = np.histogram(Counts,bins=5)
#change last bin edge to a slightly larger value, to avoid the largest element in  counts 
#overlapping with bin edges
bins[-1] = bins[-1]+1
#assigning labels to the bike counts
lb = np.digitize(Counts,bins)

#########################################################################################
#rebuilding data file
#########################################################################################
data['labels'] = lb
data = data.drop('Rented Bike Count',1)
data.to_csv('./bikes_new.csv') 
