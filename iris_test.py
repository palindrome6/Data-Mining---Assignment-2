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
    with open(os.path.join(openml_dir, "apikey_nitin.txt"), 'r') as fh:
        key = fh.readline().rstrip('\n')
    openml = APIConnector(cache_directory=cache_dir, apikey=key)
    dataset = openml.download_dataset(61)
    X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
    iris = pd.DataFrame(X, columns=attribute_names)
    iris['class'] = y
    iris.plot(kind='scatter', x='petallength', y='petalwidth', c='class', colormap='gist_rainbow', colorbar=False, s=50)
    # plt.show()
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    plot_surface(clf, X, y)
    plt.show()
