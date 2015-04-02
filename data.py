'''
Created on Mar 23, 2015

@author: aisfo
'''

from os.path import expanduser, join
from pandas.io.parsers import read_csv

DATA_ROOT = join(expanduser("~"), "data", "TELEMATICS")
MIN_DRIVER = 1
MAX_DRIVER = 3612
MIN_ROUTE = 1
MAX_ROUTE = 200

def get_route(driver_id, route_id):
    if driver_id < MIN_DRIVER or driver_id > MAX_DRIVER or route_id < MIN_ROUTE or route_id > MAX_ROUTE:
        return None
    csv_path = join(DATA_ROOT, str(driver_id), str(route_id) + ".csv")
    data = read_csv(csv_path)
    return data
 
def get_all(driver_id):   
    if driver_id < MIN_DRIVER or driver_id > MAX_DRIVER:
        return None
    routes = []
    for route_id in xrange(MAX_ROUTE):
        routes.append(get_route(driver_id, route_id+1))
    return routes
    
def save_route(driver_id, route_id, data_frame):
    if driver_id < MIN_DRIVER or driver_id > MAX_DRIVER or route_id < MIN_ROUTE or route_id > MAX_ROUTE:
        return None
    csv_path = join(DATA_ROOT, str(driver_id), str(route_id) + ".csv")
    data_frame.to_csv(csv_path, index=False)
