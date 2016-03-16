from sklearn import cross_validation
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import Bootstrap

def exercise_1():
    X, y = make_blobs(n_samples=1000, centers=2, n_features=2,
                      random_state=0)
    accuracy_lst = np.zeros([49, 2], dtype=float)
    for i in range(0,49):
        clf = KNeighborsClassifier(n_neighbors=i+1)
        scores = cross_validation.cross_val_score(
           clf, X, y, cv=10, scoring='mean_absolute_error')
        accuracy = (scores.mean(), scores.std() * 2)
        accuracy_lst[i,0] = accuracy[0]
        accuracy_lst[i,1] = accuracy[1]
    #plot
    accuracy_lst = np.around(accuracy_lst, decimals=5)
    x = np.arange(1,50, dtype=int)
    y = accuracy_lst[:,0]
    e = accuracy_lst[:,1]
    plt.style.use('ggplot')
    plt.plot(x,y, 'ro')
    plt.xticks(x, x)
    plt.margins(0.02)
    plt.show()

def exercise_2():
    X, y = make_blobs(n_samples=1000, centers=2, n_features=2,
                      random_state=0)

    accuracy_lst = np.zeros([49, 2], dtype=float)
    for i in range(0,49):
        clf = KNeighborsClassifier(n_neighbors=i+1)

        cv = cross_validation.ShuffleSplit(1000, n_iter=10,
                                test_size=0.1, random_state=0)
        scores = cross_validation.cross_val_score(
           clf, X, y, cv=cv, scoring='mean_absolute_error')
        accuracy = (scores.mean(), scores.std() * 2)
        accuracy_lst[i,0] = accuracy[0]
        accuracy_lst[i,1] = accuracy[1]
    #plot
    accuracy_lst = np.around(accuracy_lst, decimals=5)
    x = np.arange(1,50, dtype=int)
    y = accuracy_lst[:,0]
    e = accuracy_lst[:,1]
    plt.style.use('ggplot')
    plt.plot(x,y, 'ro')
    plt.xticks(x, x)
    plt.show()

if __name__ == "__main__":
    exercise_2()



