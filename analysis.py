'''
Created on Mar 24, 2015

@author: aisfo
'''

from __future__ import division

from math import sqrt, floor, log
from heapq import heappush, heappop

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.preprocessing import Imputer, MinMaxScaler


from cluster import Cluster
from features import extract_features


features = extract_features(10)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
ffeatures = imp.fit_transform(features)

stf = (ffeatures - ffeatures.mean())/ffeatures.std()

pca = PCA(2)
rstf = pca.fit_transform(stf)
tvec = sum(pca.explained_variance_ratio_)
print "pc explained variance:", pca.explained_variance_ratio_, tvec


min_max_scaler = MinMaxScaler()
rstf2 = min_max_scaler.fit_transform(rstf)


_cluster = None
pairs = []
clusters = [Cluster([pnt]) for pnt in rstf]
for i1 in xrange(len(clusters)):
    for i2 in xrange(len(clusters)):
        if i1 < i2:
            d = clusters[i1].linkage(clusters[i2])
            heappush(pairs, (d, clusters[i1], clusters[i2]))
while len(clusters) > 1:                
    closest_pair = heappop(pairs)
    first_cluster = closest_pair[1]
    second_cluster = closest_pair[2]
     
    if first_cluster.size() == 0 or second_cluster.size() == 0:
        continue
    
    if first_cluster.size() + second_cluster.size() > 100: #limit to 101?
        dists = []
        for pnt in second_cluster.points:
            dd = first_cluster.pnt_linkage(pnt)
            dists.append( (dd, pnt) )
        dists.sort()
        i = 0
        while first_cluster.size() < 101:
            first_cluster.add( dists[i][1] )
            i += 1
        
        #first_cluster.merge(second_cluster)    
        _cluster = first_cluster
        break
            
    first_cluster.merge(second_cluster)
    second_cluster.empty()

    i = 0
    while i < len(clusters):
        cluster = clusters[i]
        i = i + 1
        if cluster is second_cluster:
            i = i - 1
            del clusters[i]
        elif cluster is not first_cluster:
            d = cluster.linkage(first_cluster)
            heappush(pairs, (d, cluster, first_cluster))     

centre = _cluster.centre()
#print "the cluster centre:", centre
comp = _cluster.compactness() 
#print "the cluster compactness:", comp 

ssd = 0
dists = []
for i in xrange(len(rstf)):
    pnt = rstf[i]
    d = distance.euclidean(pnt, centre)
    ssd += d**2
    dists.append( (d, pnt) )
dists.sort()
dev = sqrt(ssd/len(rstf))
print "deviation", dev

far = dists[-1][0]
#print "the farthest point:", far

print "fr", comp/far, dev/far


fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
plot_pnts = False
lines = np.zeros((200, 6))

sum1 = dists[0][0]

final_cluster = Cluster([centre])
color = "red"
for i in xrange(0, len(dists)):
    i_dist = dists[i]
    
    sum1 += (i_dist[0] ** 2)
    
    lines[i][0] = i_dist[0]
    lines[i][1] = sqrt(sum1/(i+1))
    lines[i][2] = final_cluster.compactness()
    lines[i][3] = final_cluster.pnt_linkage(i_dist[1])
    lines[i][4] = comp
    lines[i][5] = dev
    
    
    if i > 100 and lines[i][0] > lines[i][5]:
        color = "black"
    else:   
        final_cluster.add(i_dist[1])
        color = "red"
    
    if plot_pnts:
        plt.scatter([i_dist[1][0]], [i_dist[1][1]], color=color)


if plot_pnts:
    plt.scatter([centre[0]], [centre[1]], color="green")
 
if not plot_pnts:
    plt.plot(lines[1:,0])
    plt.plot(lines[1:,1])
    plt.plot(lines[1:,2])
    plt.plot(lines[1:,3])
    plt.plot(lines[1:,4])
    plt.plot(lines[1:,5])
    
plt.show() 

    