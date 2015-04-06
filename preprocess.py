'''
Created on Mar 23, 2015

@author: aisfo
'''
from __future__ import division
from math import sqrt, atan2
import math

import pandas
import numpy as np

from data import get_all, save_route

pandas.options.mode.chained_assignment = None



def preprocess(driver):
    
    routes = get_all(driver)    
    for idx,route in enumerate(routes):
        npoints = len(route)
        route["v"] = np.zeros(npoints)
        route["a"] = np.zeros(npoints)
        route["dir"] = np.zeros(npoints)
        route["curv"] = np.zeros(npoints)
        route["v_adj"]= np.zeros(npoints)
        
        for pi in xrange(1,npoints):
            xp = route["x"][pi-1]
            x = route["x"][pi]
            yp = route["y"][pi-1]
            y = route["y"][pi]
            vp = route["v"][pi-1]
            v = sqrt((x-xp)**2 + (y-yp)**2)
            a = v - vp
            route["v"][pi] = v
            route["a"][pi] = a
            
            route["dir"][pi] = atan2((y - yp ), (x - xp))
            
            curv = route["dir"][pi] - route["dir"][pi-1]
            if curv > math.pi:
                curv -= math.pi*2 
            if curv < -math.pi:
                curv += math.pi*2 
            route["curv"][pi] = abs(curv)
    
        save_route(driver, idx+1, route)


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
    
    features = pandas.DataFrame(data=np.zeros((200, len(feature_names))), 
                         columns=feature_names)
    

    routes = get_all(driverid)
    if len(routes) != 200:
        return None
    
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


if __name__ == "__main__":
    
    st = 20
    en = 3613
    
    for i in range(st, en):
        print "preprocessing", i
        preprocess(i)
        print "done", i
    
