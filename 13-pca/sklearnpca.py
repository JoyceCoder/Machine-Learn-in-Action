import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from numpy import *

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components = 4)
X_r = pca.fit(X).transform(X)

print('各个特征的方差值占总方差的比例：',str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy','turquoise','darkorange']
lw =2

for color, i ,target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(X_r[y == i,0],X_r[y==i,1],alpha =.8,color = color,label = target_name)
    x = linspace(-3,4,50)
    #plt.plot(x,X_r[y==i,0],color = color)
plt.legend(loc = 'best',shadow=False,scatterpoints=1)
plt.title('PCA IRIS Data')
plt.show()
