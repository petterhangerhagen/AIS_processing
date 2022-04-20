""" A simple script to add distances to land in a csv file, only work with the area surrounding Trondheim """
""" Needs coastline.csv with map of zero-level elevation """

import numpy as np
from math import radians, cos, sin, asin, sqrt, pi
import pandas as pd
import warnings

warnings.simplefilter(action='ignore')

def getDisToMeterSafe(lon1, lat1, lon2, lat2, **kwarg):
    """Get distance in meters between two lon lat coords, ~15% slower than its unsafe counterpart"""
    # in case the one of the parameters is not valid this is a safe version to use
    p = pi/180
    try:
        a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
        return 12742 * asin(sqrt(a)) * 1000
    except:
        return 999999

def latlon(lon, lat):
    cp = coast.loc[(np.abs(coast['lat']-lat) <=  0.2) & (np.abs(coast['lon']-lon) <= 0.2)]
    if len(cp) == 0:
        return 30000
    cp['dist'] = cp.apply(lambda c: getDisToMeterSafe(lon,lat,c['lon'],c['lat']), axis = 1) 
    d = min(cp['dist'].tolist())
    del cp
    return min(999999, d)

coast = pd.read_csv('kyst_skjaer_dybdekurver_small.csv', sep = ';')
coast = coast.loc[coast['depth'] == 0].reset_index(drop=True)
coast = coast[['lon', 'lat']]
coast = coast.sample(n = len(coast)//10, replace = False) 

readfile = 'superPara.csv'

df = pd.read_csv(readfile, sep = ';')

df = df.loc[df['obst_speed'] >= 1]
df = df.loc[df['own_speed'] >= 1]

df = df.loc[\
        ~(((df['COLREG'] == 2.0) & (np.abs(df['own_speed']-df['obst_speed']) < 0.1)) | \
        ((df['COLREG'] == -2.0) & (np.abs(df['own_speed']-df['obst_speed']) < 0.1)))]


#df = df.loc[df['delta_course'] < 180].reset_index(drop=True)


df = df.loc[df['time'].apply(lambda s: s[-8:] < '00:30:00' and s[-8:] > '00:05:00')]


disties = []
l = len(df)
i = 1

for lon, lat in zip(df['lon_maneuver'].tolist(), df['lat_maneuver'].tolist()):
    disties.append(latlon(lon = lon, lat = lat))
    i += 1
    if i % 100 == 0:
        print("at ", i , "/", l, " coast points:", len(coast))

df['dist_land']= disties
print(df['dist_land'])

df.to_csv('paralDist.csv', sep = ';', index = False)

print(latlon(10.000043, 63.916070))


