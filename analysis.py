'''
Created on Mar 24, 2015

@author: aisfo
'''

from __future__ import division

from math import sqrt
from heapq import heappush, heappop

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer

from cluster import Cluster
from preprocess import extract_features

#returns processed route points and classifications for them
def analyze(driver):
    
    ##EXTRACT AND PROCESS FEATURES 
    print "Extracting and processing features..."
    
    #extracting features for driver
    features = extract_features(driver)
    if features is None:
        return None
    
    #impute missing values
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    ffeatures = imp.fit_transform(features)
    
    #standardize features for PCA
    stf = (ffeatures - ffeatures.mean())/ffeatures.std()
    
    #principal component analysis
    pca = PCA(2) # TODO: number of components? 
    rstf = pca.fit_transform(stf) # TODO: normalize/standardize?
    print "PCA explained variance:", pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)
    
    
    ##BUILD CORE CLUSTER AND FIND PROTOTYPE
    print "Building core cluster and finding prototype..."
    
    #agglomerative clustering until cluster with 101 elements is build
    core_cluster = None
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
 
            core_cluster = first_cluster
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
    
    #average of core cluster is prototype
    prototype = core_cluster.centre()
    print "Prototype:", prototype
    
    
    ##CLASSIFY THE ROUTES
    print "Classifying the routes..."
    
    #sort the routes by distance from the prototype
    sorted_pnts = []
    for i in xrange(len(rstf)):
        pnt = rstf[i]
        d = distance.euclidean(pnt, prototype)
        sorted_pnts.append( (d, pnt, i) )
    sorted_pnts.sort()
    
    #compute compactness and radius of core cluster
    comp = core_cluster.compactness() 
    rad = core_cluster.radius()
    
    #use core cluster compactness to farthest point distance measure 
    #for evaluating the feature selection
    far = sorted_pnts[-1][0]    
    print "Compactness to Farthest Point:", comp/far
    
    #plotting
    lines = np.zeros((200, 7))
    
    #sum of squared distances
    ssum = sorted_pnts[0][0] ** 2
    #final cluster with prototype included 
    final_cluster = Cluster([prototype])
    #all zeros labels array
    labels = np.zeros(200)
    
    #classify as true until cutoff
    for i in xrange(0, len(sorted_pnts)):
        point_tuple = sorted_pnts[i]
        dist = point_tuple[0]
        point = point_tuple[1]
        l_idx = point_tuple[2]
        
        ssum += (dist ** 2)
        
        lines[i][0] = dist
        lines[i][1] = sqrt(ssum/(i+1))
        lines[i][2] = final_cluster.compactness()
        lines[i][3] = final_cluster.min_linkage(point)
        lines[i][4] = final_cluster.pnt_linkage(point)
        lines[i][5] = comp
        lines[i][6] = rad

        if lines[i][3] < lines[i][6]: #TODO: Choice and evaluation of cutoff
            labels[l_idx] = 1
            final_cluster.add(point)
            
    plt.plot(lines[1:,0])
    plt.plot(lines[1:,1])
    plt.plot(lines[1:,2])
    plt.plot(lines[1:,3])
    plt.plot(lines[1:,4])
    plt.plot(lines[1:,5])
    plt.plot(lines[1:,6])
        
    return rstf, labels
    
    

if __name__ == "__main__":
    
    plt.figure(1)
    
    points, labels = analyze(1)
    
    plt.figure(2)
    colors = ["red" if l == 1 else "black" for l in labels]
    plt.scatter( points[:,0], points[:,1], color=colors)

    plt.show()
    