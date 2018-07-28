from sklearn.cluster import KMeans
from numpy import *
import kMeans

X=kMeans.loadDataSet('testSet.txt')
kmeans = KMeans(n_clusters=4,random_state=0).fit(X)
print("sklearn实现质心列表为：",kmeans.cluster_centers_)
centroids, _ =kMeans.kMeans(mat(X),4)
print("python实现质心列表为：",centroids)
