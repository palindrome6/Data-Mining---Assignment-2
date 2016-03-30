print(__doc__)
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import math
# import some data to play with
X, y = make_blobs(n_samples=1000, centers=2, n_features=2,
                      random_state=0)

h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C1 = math.pow(2,-15)  # SVM regularization parameter
A1 = math.pow(2,-15)  # SVM  parameter
svc1 = svm.SVC(kernel='linear', gamma=A1, C=C1).fit(X, y)
rbf_svc1 = svm.SVC(kernel='rbf', C=C1).fit(X, y)
poly_svc1 = svm.SVC(kernel='poly',  gamma=A1, C=C1).fit(X, y)
# lin_svc = svm.LinearSVC(C=C).fit(X, y)


C2 = math.pow(2, 15)  # SVM regularization parameter
A2 = math.pow(2,15)  # SVM  parameter
svc2= svm.SVC(kernel='linear',  gamma=A2, C=C2).fit(X, y)
rbf_svc2 = svm.SVC(kernel='rbf', C=C2).fit(X, y)
poly_svc2 = svm.SVC(kernel='poly',  gamma=A2, C=C2).fit(X, y)





# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel(C1)',
          'SVC with RBF kernel (C1)',
          'SVC with polynomial kernel (C1)',
          'SVC with linear kernel(C2)',
          'SVC with RBF kernel (C2)',
          'SVC with polynomial kernel (C2)']


for i, clf in enumerate((svc1, rbf_svc1, poly_svc1, svc2, rbf_svc2, poly_svc2)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(2, 3, i + 1)
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