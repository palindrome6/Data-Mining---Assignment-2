from openml.apiconnector import APIConnector
import pandas as pd
import os
import difflib
import subprocess
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, show
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
# from sklearn.cross_validation import Bootstrap
from sklearn.tree import DecisionTreeClassifier
from scipy import spatial
from sklearn import svm
from sklearn import metrics

def problem_linear():
    X, y = make_blobs(n_samples=1000, centers=2, n_features=2,
                      random_state=0)

    clf = svm.SVC(kernel='linear')
    clf.fit(X,y)

    A = clf.predict(X)
    B = y
    sim = result = 1 - spatial.distance.cosine(A, B)
    print sim*100
    # plt.figure()
    # plt.scatter(X[:,0],X[:,1], c=A)
    # plt.show()
    # show(block=False)
    print A
    print B

    scores = cross_validation.cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
    print scores.mean()




def problem_poly():
    X, y = make_blobs(n_samples=1000, centers=2, n_features=2,
                      random_state=0)

    clf = svm.SVC(kernel='poly')
    clf.fit(X,y)

    A = clf.predict(X)
    B = y
    sim = result = 1 - spatial.distance.cosine(A, B)
    print sim*100
    plt.figure()
    plt.scatter(X[:,0],X[:,1], c=A)
    plt.show()
    show(block=False)


def problem_rbf():
    X, y = make_blobs(n_samples=1000, centers=2, n_features=2,
                      random_state=0)

    clf = svm.SVC(kernel='rbf')
    clf.fit(X,y)

    A = clf.predict(X)
    B = y
    sim = result = 1 - spatial.distance.cosine(A, B)
    print sim*100
    plt.figure()
    plt.scatter(X[:,0],X[:,1], c=A)
    plt.show()
    show(block=False)


if __name__ == "__main__":
    problem_linear()
    # problem_poly()
    # problem_rbf()
