from openml.apiconnector import APIConnector
import pandas as pd
import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.datasets import load_iris
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn import tree
import pydot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

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
    iris['class_type'] = y


    name_lst = list(iris.columns.values)
    # print name_lst
    # print iris[:2]
    # print iris.ix[:5,:18]

    #A
    # plot_histogram()


    #B
    data_binary = iris[iris.class_type != 3]
    data_binary = data_binary[data_binary.class_type != 0]

    # print data_binary.ix[:,[0,1]]

    #C
    training = data_binary.values[:, :18]
    classes = map(list, data_binary.values[:,18:])
    classes = np.array(classes).astype(int).flatten()
    clf = DecisionTreeClassifier()
    clf.fit(training, classes)
    dlf = DecisionTreeClassifier()
    dlf.fit(training, classes)


print "CART"
print("Training error =", zero_one_loss(classes, clf.predict(training)))


X_train, X_test, y_train, y_test = train_test_split(training, classes)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("Training error =", zero_one_loss(y_train, clf.predict(X_train)))
print("Test error =", zero_one_loss(y_test, clf.predict(X_test)))

scores = []
print "K-fold cross validation"
for train, test in KFold(n=len(training), n_folds=5, random_state=42):
    X_train, y_train = training[train], classes[train]
    X_test, y_test = training[test], classes[test]
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    scores.append(zero_one_loss(y_test, clf.predict(X_test)))
#
print("CV error = %f +-%f" % (np.mean(scores), np.std(scores)))
#
print "Cross validation"
scores = cross_val_score(DecisionTreeClassifier(), training, classes,
                         cv=KFold(n=len(training), n_folds=5, random_state=42),
                         scoring="accuracy")
print("CV error = %f +-%f" % (1. - np.mean(scores), np.std(scores)))
print("Accuracy =", accuracy_score(y_test, clf.predict(X_test)))
print("Precision =", precision_score(y_test, clf.predict(X_test)))
print("Recall =", recall_score(y_test, clf.predict(X_test)))
print("F =", fbeta_score(y_test, clf.predict(X_test), beta=1))
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

print "Random forest classifier"
tlf = RandomForestClassifier(n_estimators=500)
tlf.fit(training, classes)

print("Training error =", zero_one_loss(classes, tlf.predict(training)))


X_train, X_test, y_train, y_test = train_test_split(training, classes)
tlf = RandomForestClassifier()
tlf.fit(X_train, y_train)
print("Training error =", zero_one_loss(y_train, tlf.predict(X_train)))
print("Test error =", zero_one_loss(y_test, tlf.predict(X_test)))

scores = []
print "K-fold cross validation"
for train, test in KFold(n=len(training), n_folds=5, random_state=42):
    X_train, y_train = training[train], classes[train]
    X_test, y_test = training[test], classes[test]
    tlf = RandomForestClassifier().fit(X_train, y_train)
    scores.append(zero_one_loss(y_test, tlf.predict(X_test)))
#
print("CV error = %f +-%f" % (np.mean(scores), np.std(scores)))
#
print "Cross validation"
scores = cross_val_score(RandomForestClassifier(), training, classes,
                         cv=KFold(n=len(training), n_folds=5, random_state=42),
                         scoring="accuracy")
print("CV error = %f +-%f" % (1. - np.mean(scores), np.std(scores)))
print("Accuracy =", accuracy_score(y_test, tlf.predict(X_test)))
print("Precision =", precision_score(y_test, tlf.predict(X_test)))
print("Recall =", recall_score(y_test, tlf.predict(X_test)))
print("F =", fbeta_score(y_test, tlf.predict(X_test), beta=1))
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

print "Extra Tree classifier"
rlf = ExtraTreeClassifier()
rlf.fit(training, classes)

print("Training error =", zero_one_loss(classes, rlf.predict(training)))


X_train, X_test, y_train, y_test = train_test_split(training, classes)
rlf = ExtraTreeClassifier()
rlf.fit(X_train, y_train)
print("Training error =", zero_one_loss(y_train, rlf.predict(X_train)))
print("Test error =", zero_one_loss(y_test, rlf.predict(X_test)))

scores = []
print "K-fold cross validation"
for train, test in KFold(n=len(training), n_folds=5, random_state=42):
    X_train, y_train = training[train], classes[train]
    X_test, y_test = training[test], classes[test]
    rlf = ExtraTreeClassifier().fit(X_train, y_train)
    scores.append(zero_one_loss(y_test, rlf.predict(X_test)))
#
print("CV error = %f +-%f" % (np.mean(scores), np.std(scores)))
#
print "Cross validation"
scores = cross_val_score(ExtraTreeClassifier(), training, classes,
                         cv=KFold(n=len(training), n_folds=5, random_state=42),
                         scoring="accuracy")
print("CV error = %f +-%f" % (1. - np.mean(scores), np.std(scores)))
print("Accuracy =", accuracy_score(y_test, rlf.predict(X_test)))
print("Precision =", precision_score(y_test, rlf.predict(X_test)))
print("Recall =", recall_score(y_test, rlf.predict(X_test)))
print("F =", fbeta_score(y_test, rlf.predict(X_test), beta=1))


print "Feature importance, dlf:"
print(dlf.feature_importances_)





# Build a classification task using 3 informative features


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier()

forest.fit(training, classes)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print name_lst[f]
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()