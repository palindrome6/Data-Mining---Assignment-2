from openml.apiconnector import APIConnector
import pandas as pd
import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier


def plot_histogram():
    class_histogram = iris['class_type']
    print class_histogram.value_counts()
    alphab = [1,2,3,0]
    frequencies = [81, 61, 4, 2]
    pos = np.arange(len(alphab))
    width = 1.0     # gives histogram aspect to the bar diagram
    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(alphab)
    plt.bar(pos, frequencies, width, color='#009999')
    plt.show()

def plot_surface(clf, X, y,
                 xlim=(-10, 10), ylim=(-10, 10), n_steps=250,
                 subplot=None, show=True):
    if subplot is None:
        fig = plt.figure()
    else:
        plt.subplot(*subplot)

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n_steps),
                         np.linspace(ylim[0], ylim[1], n_steps))

    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.RdBu_r)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    if show:
        plt.show()


if __name__ == "__main__":
    home_dir = os.path.expanduser("~")
    openml_dir = os.path.join(home_dir, ".openml")
    cache_dir = os.path.join(openml_dir, "cache")
    with open(os.path.join(openml_dir, "apikey.txt"), 'r') as fh:
        key = fh.readline().rstrip('\n')
    openml = APIConnector(cache_directory=cache_dir, apikey=key)
    dataset = openml.download_dataset(10)
    X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
    iris = pd.DataFrame(X, columns=attribute_names)
    iris['class'] = y


    #A
    # plot_histogram()


    #B



    #C
    data_binary = iris[:]
    training = data_binary.values[:,[0,1]]
    classes = map(list, data_binary.values[:,18:])
    classes = np.array(classes).astype(int).flatten()
    clf = DecisionTreeClassifier()
    clf.fit(training, classes)
    plot_surface(clf, training, classes)





    # from sklearn.datasets import load_iris
    # from sklearn.tree import DecisionTreeClassifier
    #
    # # Parameters
    # n_classes = 3
    # plot_colors = "bry"
    # plot_step = 0.02
    #
    # # Load data
    # iris = load_iris()
    #
    # # print iris
    # for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
    #                                 [1, 2], [1, 3], [2, 3]]):
    #     # We only take the two corresponding features
    #     X = iris.data[:, pair]
    #     print pairidx
    #     # print X
    #     y = iris.target
    #     # print y
    #
    #     # Shuffle
    #     idx = np.arange(X.shape[0])
    #     np.random.seed(13)
    #     np.random.shuffle(idx)
    #     X = X[idx]
    #     y = y[idx]
    #
    #     # Standardize
    #     mean = X.mean(axis=0)
    #     std = X.std(axis=0)
    #     X = (X - mean) / std
    #
    #     # Train
    #     clf = DecisionTreeClassifier().fit(X, y)
    #
    #     # Plot the decision boundary
    #     plt.subplot(2, 3, pairidx + 1)
    #
    #     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
    #                          np.arange(y_min, y_max, plot_step))
    #
    #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)
    #     cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    #
    #     plt.xlabel(iris.feature_names[pair[0]])
    #     plt.ylabel(iris.feature_names[pair[1]])
    #     plt.axis("tight")
    #
    #     # Plot the training points
    #     for i, color in zip(range(n_classes), plot_colors):
    #         idx = np.where(y == i)
    #         plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
    #                     cmap=plt.cm.Paired)
    #
    #     plt.axis("tight")
    #
    # plt.suptitle("Decision surface of a decision tree using paired features")
    # plt.legend()
    # plt.show()
    #
    #
    #
    #
    #
    #
    #
    #
    #
