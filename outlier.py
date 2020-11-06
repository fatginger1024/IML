import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt


data = pd.read_csv('./bikes_seoul.csv')
X = np.c_[data['Visibility (10m)'],data['Rented Bike Count']]


# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=500, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(X)
X_scores = clf.negative_outlier_factor_
plt.figure(figsize=(7,5))
plt.title("Local Outlier Factor (LOF)")
plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points',alpha=.1)
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(X[:, 0], X[:, 1], s=200 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores',alpha=.3)
plt.axis('tight')
plt.xlabel("Visibility(10m)")
plt.ylabel("Rented Bike Count")
legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.savefig('./plots/outlier.eps')
plt.show()
