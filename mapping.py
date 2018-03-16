import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.load('feature_125_1000.npy')
# print(X.shape)
y = np.load('name_125_1000.npy')
y = [int(a.split('_')[-1].split('.')[0]) for a in y]
all_id_in_y = np.unique(y)
# print(y)
y = [np.where(all_id_in_y==a)[0].tolist()[0] for a in y]
y = np.array(y)

# print(y)

# print(y2)
# print(all_id_in_y)

# clf = lda(n_components=2)
# x_new = clf.fit_transform(X, y)

# # pca = PCA(n_components=2)
# # x_new = pca.fit_transform(X, y)

# plt.scatter(x_new[:, 0], x_new[:, 1], c=y)
# plt.show()

# 3d
clf = lda(n_components=3)
x_new = clf.fit_transform(X, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_new[:, 0], x_new[:, 1], x_new[:, 2], c=y, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()