import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from scipy.spatial import ConvexHull

threshold = 0.5
y_true = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0])
scores_a = np.array([[1,1,0,0,1,1,0,0,1,0,0,0,0]])
scores_b = np.array([[1,1,1,1,0,1,1,0,1,0,1,0,0]])
scores_c = np.array([[0.8,0.9,0.7,0.6,0.4,0.8,0.4,0.4,0.6,0.4, 0.4, 0.4,0.2]])
np.place(scores_c, scores_c > threshold, 1)
np.place(scores_c, scores_c <= threshold, 0)
scores = np.append(scores_a, scores_b, axis=0)
scores = np.append(scores, scores_c, axis=0)
n_classes = 3


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true, scores[i], pos_label=1)
    roc_auc[i] = auc(fpr[i], tpr[i])

print fpr[0]
print tpr[0]
#plot for each class
names=['A','B','C']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.4f})'
                                   ''.format(names[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves')
plt.legend(loc="lower right")
plt.show()




#convexHull


