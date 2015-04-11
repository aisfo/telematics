'''
Created on Mar 24, 2015

@author: aisfo
'''
from __future__ import division

from heapq import heappush, heappop, heapify
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer

from cluster import Cluster
from preprocess import extract_features


#returns processed route points and classifications for them
def analyze(driver):
    
    ### EXTRACT AND PROCESS FEATURES 
    print "Extracting and processing features..."
    
    #extracting features for driver
    features = extract_features(driver)
    if features is None:
        return None
    
    #impute missing values
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    features = imp.fit_transform(features)

    #standardize features for PCA
    features = (features - features.mean())/features.std() #TODO: standardize?
    
    #principal component analysis
    pca = PCA(0.8) #TODO: number of components? 
    features = pca.fit_transform(features)
    print "PCA explained variance:", pca.explained_variance_ratio_, pca.n_components_   
    
    
    ### BUILD CORE CLUSTER
    print "Building core cluster and finding prototype..."
    
    #agglomerative clustering until cluster with 101 elements is build
    core_cluster = None
    pairs = []
    clusters = [Cluster([pnt]) for pnt in features]
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
        
        if first_cluster.size() + second_cluster.size() > 100:
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
    
    
    ### CLASSIFY THE ROUTES
    print "Classifying the routes..."
    
    #all zeros labels array
    labels = np.zeros(200)
    
    #sort the routes by distance from the prototype
    sorted_pnts = []
    for i in xrange(len(features)):
        pnt = features[i]
        closest, dist = core_cluster.closest_point(pnt)
        if dist == 0:
            labels[i] = 1
        else:
            sorted_pnts.append( (dist, i, pnt, closest) )
    sorted_pnts.sort()
    
    #compute compactness and radius of core cluster
    core_cluster_comp = core_cluster.compactness() 
    core_cluster_diam = core_cluster.diameter()        
    
    #use core cluster compactness to farthest point distance measure 
    #for evaluating the feature selection
    far = sorted_pnts[-1][0]
    print "Compactness to Farthest Point:", core_cluster_comp/far, "[minimize]"

    #classify points based on density
    heapify(sorted_pnts)
    while len(sorted_pnts) > 0:
        candidate = heappop(sorted_pnts)
        point = candidate[2]
        dist = candidate[0]
        idx = candidate[1]
        closest = candidate[3]
        
        nbrs = core_cluster.neighbours(closest, core_cluster_diam)
        if len(nbrs) >= 100 and dist < core_cluster_comp:
            core_cluster.add(point)
            labels[idx] = 1
            
            tsorted_pnts = []
            for sp in sorted_pnts:
                sp_pnt = sp[2]
                sp_idx = sp[1]
                sp_closest, sp_dist = core_cluster.closest_point(sp_pnt)
                tsorted_pnts.append( (sp_dist, sp_idx, sp_pnt, sp_closest) )
            sorted_pnts = tsorted_pnts
            heapify(sorted_pnts)
        
        
    return features, labels
    
    

if __name__ == "__main__":
    
    points, labels = analyze(333)
    
    plt.figure(1)
    colors = ["red" if l == 1 else "black" for l in labels]
    plt.scatter( points[:,0], points[:,1], color=colors)
    print "True", sum(labels)

    #plt.show()
    