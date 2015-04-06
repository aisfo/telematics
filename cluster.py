'''
Created on Mar 24, 2015

@author: aisfo
'''
from __future__ import division
import numpy as np
from scipy.spatial import distance

class Cluster(): 
    
    def __init__(self, pnts):
        self.points = np.array(pnts)
        
    def linkage(self, other):
        dist_sum = []
        for p1 in self.points:
            for p2 in other.points:
                dist_sum.append( distance.euclidean(p1, p2) )
        return np.mean(dist_sum)

    def pnt_linkage(self, pnt):
        dist_sum = []
        for p1 in self.points:
            dist_sum.append( distance.euclidean(p1, pnt) )
        return np.mean(dist_sum)

    def min_linkage(self, pnt):
        dist_sum = []
        for p1 in self.points:
            dist_sum.append( distance.euclidean(p1, pnt) )
        return np.min(dist_sum)
    
    def merge(self, other):
        self.points = np.append(self.points, other.points, 0)
    
    def size(self):
        return len(self.points)

    def centre(self):
        return self.points.mean(0)
    
    def add(self, elt):
        self.points = np.append(self.points, [elt], 0)
    
    def empty(self):
        self.points = np.array([])

    def __cmp__(self, other):
        if other is not self:
            return 1
        else:
            return 0
        
    def compactness(self):
        l = self.size()
        dists = []
        for i in xrange(l):
            for j in xrange(i+1, l):
                p1 = self.points[i]
                p2 = self.points[j]
                dists.append(distance.euclidean(p1, p2))
        if len(dists) == 0:
            return 0
        return np.mean(dists)
    
    def radius(self):
        l = self.size()
        dists = []
        for i in xrange(l):
            for j in xrange(i+1, l):
                p1 = self.points[i]
                p2 = self.points[j]
                dists.append(distance.euclidean(p1, p2))
        if len(dists) == 0:
            return 0
        return np.max(dists)