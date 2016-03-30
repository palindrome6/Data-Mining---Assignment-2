# Global imports and settings
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy.spatial import ConvexHull
from openml.apiconnector import APIConnector
import pandas as pd
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs
import math

X, y = make_blobs(n_samples=1000, centers=2, n_features=2,
                      random_state=0)
h = .02
clf = svm.SVC(kernel='rbf')
param_dist1 = dict(C=[math.pow(2,-15)])
param_dist2 = dict(C=[math.pow(2,-5)])
param_dist3 = dict(C=[math.pow(2,5)])
param_dist4 = dict(C=[math.pow(2,15)])
param_dist5 = dict(gamma=[math.pow(2,-15)])
param_dist6 = dict(gamma=[math.pow(2,-5)])
param_dist7 = dict(gamma=[math.pow(2,5)])
param_dist8 = dict(gamma=[math.pow(2,15)])
grid1= GridSearchCV(clf, param_dist1, cv=10, scoring="roc_auc")
grid2= GridSearchCV(clf, param_dist2, cv=10, scoring="roc_auc")
grid3= GridSearchCV(clf, param_dist3, cv=10, scoring="roc_auc")
grid4= GridSearchCV(clf, param_dist4, cv=10, scoring="roc_auc")
grid5= GridSearchCV(clf, param_dist5, cv=10, scoring="roc_auc")
grid6= GridSearchCV(clf, param_dist6, cv=10, scoring="roc_auc")
grid7= GridSearchCV(clf, param_dist7, cv=10, scoring="roc_auc")
grid8= GridSearchCV(clf, param_dist8, cv=10, scoring="roc_auc")
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with RBF kernel (C=2^-15)','SVC with RBF kernel (C=2^-5)',
          'SVC with RBF kernel (C=2^5)','SVC with RBF kernel (C=2^15)',
          'SVC with RBF kernel (gamma=2^-15)','SVC with RBF kernel (gamma=2^-5)',
          'SVC with RBF kernel (gamma=2^5)','SVC with RBF kernel (gamma=2^15)']


for i, clf in enumerate((grid1,grid2,grid3,grid4,grid5,grid6,grid7,grid8)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(4, 2, i + 1)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()