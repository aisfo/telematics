'''
Created on Mar 23, 2015

@author: aisfo
'''
from __future__ import division
import numpy as np
from math import sqrt, atan2
import math
from data import get_all, save_route
import pandas

pandas.options.mode.chained_assignment = None

driver = 10

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
