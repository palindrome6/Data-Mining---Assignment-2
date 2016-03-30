from openml.apiconnector import APIConnector
import os
from sklearn.datasets.samples_generator import make_blobs
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from scipy import spatial
from scipy.stats import randint as sp_randint
from sklearn import svm
import math





def exercise():
    apikey = 'fbc6d4b7868ce52640f6ec74cf076f48'
    connector = APIConnector(apikey=apikey)
    #loading data
    dataset = connector.download_dataset(59)
    X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
    # iris = pd.DataFrame(X, columns=attribute_names)

    clf = svm.SVC(kernel='rbf')
    # gammapar = []
    # for i in range(-15, 16, 1):
    #     gammapar.append(math.pow(2,i));
    # param_dist = dict(gamma=gammapar)
    # print gammapar
    r = np.logspace(-15, 15, 10, base=2)
    param_dist = {'gamma': r}
    rand = GridSearchCV(clf, param_dist, cv=10, scoring="roc_auc")

    rand.fit(X,y)
    rand.grid_scores_
    rand_mean_scores =[result.mean_validation_score for result in rand.grid_scores_]
    print rand.best_score_
    print rand.best_params_

    plt.style.use('ggplot')

    # x_labels = [i for i in range(31)]
    # gammapar1 = []
    # for i in range(-15, 16, 1):
    #     temp = "2^"+str(i)
    #     gammapar1.append(temp);
    # plt.plot(x_labels, rand_mean_scores)
    # plt.xticks(x_labels, gammapar1 )
    # plt.xlabel('Gamma')
    # plt.ylabel('AUC')
    # plt.show()
    #
    x_labels = [i for i in range(10)]
    gammapar1 = []
    for i in range(11):
        temp = r[i-1]
        gammapar1.append(temp);
    # plt.plot(x_labels, rand_mean_scores)
    # plt.xticks(x_labels, gammapar1 )
    # plt.xlabel('Gamma')
    # plt.ylabel('AUC')
    # plt.show()
    print rand_mean_scores
    print r
    print x_labels
    print gammapar1


if __name__ == "__main__":
    exercise()
