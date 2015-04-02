'''
Created on Mar 23, 2015

@author: aisfo
'''
from __future__ import division
import numpy as np
from pandas import DataFrame
import math


from data import get_all


def extract_features(driverid):
    
    nparts = 11
    mangle = (math.pi/2)/nparts
    
    
    feature_names = []
    for p in xrange(nparts):
        #feature_names.append("v{}".format(p))
        #feature_names.append("v{}d".format(p))
        feature_names.append("a{}".format(p))
        feature_names.append("a{}d".format(p))
        feature_names.append("a{}p".format(p))
    
    features = DataFrame(data=np.zeros((200, len(feature_names))), 
                         columns=feature_names)
    

    routes = get_all(driverid)
    for idx,route in enumerate(routes):
         
        vel = [ [] for p in xrange(nparts) ]
        acc = [ [] for p in xrange(nparts) ]
        accp = [ [] for p in xrange(nparts) ]
        
        for i in xrange(len(route)):
            curv = route["curv"][i]
            v = route["v"][i]
            a = route["a"][i]
           
            for p in xrange(nparts):
                if mangle*p <= curv and (curv < mangle*(p+1) or (p+1) == nparts):
                    vel[p].append(v)
                    acc[p].append(a)
                    if i >= 10:
                        accp[p].append(np.mean(route["a"][i-10:i]))
                    
                
        for p in xrange(nparts):
            #features["v{}".format(p)][idx] = np.mean(vel[p])
            #features["v{}d".format(p)][idx] = np.std(vel[p])
            features["a{}".format(p)][idx] = np.mean(acc[p])
            features["a{}d".format(p)][idx] = np.std(acc[p])
            features["a{}p".format(p)][idx] = np.mean(accp[p])
            
    
    return features
    
    
    
