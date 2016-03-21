from sklearn import cross_validation
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, Bootstrap, ShuffleSplit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import zero_one_loss, accuracy_score, mean_squared_error

def exercise_1():
    X, y = make_blobs(n_samples=1000,centers=50, n_features=2, random_state=0)
    n_samples = len(X)
    kf = cross_validation.KFold(n_samples, n_folds=10, shuffle=False, random_state=None)
    # kf = cross_validation.ShuffleSplit(1000,n_iter=25, test_size=0.1, train_size=0.9, random_state=None)

    error_total = np.zeros([49, 1], dtype=float)
    for k in range(1,50):
        error = []
        clf = KNeighborsClassifier(n_neighbors=k)
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            error.append( zero_one_loss(y_test, clf.predict(X_test)) )


            # error.append(clf.predict(X_test))
            # error.append( 1. - clf.score(X_test, y_test) ) #, accuracy_score(y_test, clf.predict(X_test))
            # error.append(mean_squared_error(y_test, clf.predict(X_test)))
            # error.append()
        # print error
        error_total[k-1, 0] = np.array(error).mean()
    # print error_total
    x = np.arange(1,50, dtype=int)
    plt.style.use('ggplot')
    plt.plot(x, error_total[:, 0], '#009999', marker='o')
    # plt.errorbar(x, accuracy_lst[:, 0], accuracy_lst[:, 1], linestyle='None', marker='^')
    plt.xticks(x, x)
    plt.margins(0.02)
    plt.xlabel('K values')
    plt.ylabel('Missclasification Error')
    plt.show()
    # print cross_validation.cross_val_score(KNeighborsClassifier(n_neighbors=3), X, y, cv=cv, scoring='zero_one_loss')
    # print cross_validation.cross_val_score(KNeighborsClassifier(n_neighbors=3), X, y, cv=cv, scoring='accuracy')


def exercise_2a():
    X, y = make_blobs(n_samples=1000,centers=50, n_features=2, random_state=0)
    # plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    # plt.show()
    kf = KFold(1000, n_folds=10, shuffle=False, random_state=None)
    accuracy_lst = np.zeros([49, 2], dtype=float)
    accuracy_current = np.zeros(10, dtype=float)
    for k in range(1,50):
        iterator = 0
        clf = KNeighborsClassifier(n_neighbors=k)
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            accuracy_current[iterator] = (1. - clf.score(X_test,y_test))
            iterator+=1
        accuracy_lst[k-1, 0] = accuracy_current.mean()
        # accuracy_lst[k-1, 1] = accuracy_current.std() #confidence interval 95%
    x = np.arange(1,50, dtype=int)
    plt.style.use('ggplot')
    plt.plot(x, accuracy_lst[:, 0], '#009999', marker='o')
    # plt.errorbar(x, accuracy_lst[:, 0], accuracy_lst[:, 1], linestyle='None', marker='^')
    plt.xticks(x, x)
    plt.margins(0.02)
    plt.xlabel('K values')
    plt.ylabel('Missclasification Error')
    plt.show()

def exercise_2b():
    X, y = make_blobs(n_samples=1000,centers=50, n_features=2, random_state=0)
    kf = ShuffleSplit(100, train_size= 0.9, test_size=0.1, random_state=0)
    # kf = KFold(1000, n_folds=10, shuffle=False, random_state=None)
    accuracy_lst = np.zeros([49, 2], dtype=float)
    accuracy_current = np.zeros(10, dtype=float)
    for k in range(1,50):
        iterator = 0
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train, y_train)
            accuracy_current[iterator] = (1. - clf.score(X_test,y_test))
            iterator+=1
            print mean_squared_error(y_test, clf.predict(X_test))
        accuracy_lst[k-1, 0] = accuracy_current.mean()
        accuracy_lst[k-1, 1] = accuracy_current.var()#*2 #confidence interval 95%
    x = np.arange(1,50, dtype=int)
    plt.style.use('ggplot')
    plt.plot(x, accuracy_lst[:, 1], '#009999', marker='o')
    # plt.errorbar(x, accuracy_lst[:, 0], accuracy_lst[:, 1], linestyle='None', marker='^')
    plt.xticks(x, x)
    plt.margins(0.02)
    plt.xlabel('K')
    plt.ylabel('Variance')
    plt.show()



if __name__ == "__main__":
    # exercise_2a()
    exercise_1()


    # X, y = make_blobs(n_samples=1000,centers=50, n_features=2, random_state=0)
    # plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    # plt.show()
    # folds = KFold(1000, n_folds=10, shuffle=False, random_state=None)
    # missClassTotal = []
    # for k in range(1,50):
    #     yay = BaggingClassifier(KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', p=2, metric='minkowski'),max_samples=900, n_estimators=100, max_features=1)
    #     yay.fit(X,y)
    #     missClassTotal.append(1-yay.score(X,y))
    # plt.plot(list(range(1,50)), missClassTotal)
    # plt.axis([1, 49, 0, 1])
    # plt.xlabel('k', fontsize=14, color='black')
    # plt.ylabel('misclassification rate', fontsize=14, color='black')
    # plt.show()


