from openml.apiconnector import APIConnector
import pandas as pd
import os
import numpy as np
import matplotlib.mlab as mlab
import pydot
import matplotlib.pyplot as plt
from sklearn import tree
from IPython.display import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals.six import StringIO
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

# Helper functions
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
    dataset = openml.download_dataset(10)
    X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
    iris = pd.DataFrame(X, columns=attribute_names)
    iris['class_type'] = y
    print len(iris)
    data_binary = iris[iris.class_type != 3]
    data_binary = data_binary[data_binary.class_type != 0]
    print len(data_binary)

    n_sample = data_binary['class_type']
    # print data_binary['lymphatics'][:3]
    training = data_binary.values[:,:18]
    print
    classes = data_binary.values[:,18:]
    # attr_names = data_binary.columns.values)[0:18]
    #rodr was here
    print len(training)
    print len(classes)
    clf = DecisionTreeClassifier()
    clf = clf.fit(training, classes)
    print clf
    # comment








