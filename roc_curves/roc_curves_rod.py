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




n_classes = 5
y_true = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0])
scores_a = np.array([[1,1,0,0,1,1,0,0,1,0,0,0,0]])
scores_b = np.array([[1,1,1,1,0,1,1,0,1,0,1,0,0]])
scores_c1 = np.array([[0.8,0.9,0.7,0.6,0.4,0.8,0.4,0.4,0.6,0.4, 0.4, 0.4,0.2]])
scores_c2 = np.array([[0.8,0.9,0.7,0.6,0.4,0.8,0.4,0.4,0.6,0.4, 0.4, 0.4,0.2]])
scores_c3 = np.array([[0.8,0.9,0.7,0.6,0.4,0.8,0.4,0.4,0.6,0.4, 0.4, 0.4,0.2]])

scores_c1 = np.where(scores_c1>0.5,1,0)
scores_c2 = np.where(scores_c2>0.3,1,0)
scores_c3 = np.where(scores_c3>0.6,1,0)


scores  = np.concatenate((scores_a,scores_b,scores_c1, scores_c2, scores_c3), axis=0)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true, scores[i], pos_label=1)
    roc_auc[i] = auc(fpr[i], tpr[i])

# print fpr[1]
#plot for each class
plt.style.use('ggplot')
names = ['A','B','C1_0.5', 'C2_0.3', 'C3_0.6']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.4f})'
                                   ''.format(names[i], roc_auc[i]), linewidth=2.0)
# plt.plot([0, 1], [0, 1], 'k')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves')
plt.legend(loc="lower right")
# plt.show()

print 0.85714286- (.2*0.33333333)
print fpr[1]
print tpr[1]

# #convexHull
points = np.vstack((fpr[0], tpr[0])).T
for i in range(n_classes):
    points = np.append(points, np.vstack((fpr[i], tpr[i])).T, axis=0)
hull = ConvexHull(points)


iterpoint = iter(hull.simplices)
next(iterpoint)
for simplex in iterpoint:
    plt.plot(points[simplex, 0], points[simplex, 1], '#ff3300', lw=5, alpha=0.5)

plt.plot(points[:,0], points[:,1], 'o', color = '#7a7a52', alpha=0.5)
plt.plot([0,1],[0,1], 'k--')
#margins
plot_margin = 0.02
x0, x1, y0, y1 = plt.axis()
plt.axis((x0 - plot_margin,
          x1 + plot_margin,
          y0 - plot_margin,
          y1 + plot_margin))
plt.show()

