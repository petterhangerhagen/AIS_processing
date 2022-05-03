"""Requirements:"""
import datetime
import numpy as np


def rotate(vec, ang):
    """
    :param vec: 2D vector
    :param ang: angle in radians
    :return: input vector rotated by the input angle
    """
    r_mat = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    rot_vec = np.dot(r_mat, vec)
    return rot_vec


def normalize_2pi(angle):
    """
    :param angle: Angle in radians
    :type angle: float
    :return: Angle in radians normalized to [0, 2*pi)
    """

    if np.isnan(angle):
        return np.nan
    else:
        while angle >= 2 * np.pi:
            angle -= 2 * np.pi
        while angle < 0:
            angle += 2 * np.pi
        return angle


def normalize_pi(angle):
    """
    :param angle: Angle in radians
    :type angle: float
    :return: Angle in radians normalized to [-pi, pi)
    """

    if np.isnan(angle):
        return np.nan
    else:
        while angle >= np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def normalize_360_deg(angle):
    """
    :param angle: Angle in degrees
    :type angle: float
    :return: Angle in degrees normalized to [0, 360)
    """

    if np.isnan(angle):
        return np.nan
    else:
        while angle >= 360:
            angle -= 360
        while angle < 0:
            angle += 360
        return angle


def normalize_180_deg(angle):
    """
    :param angle: Angle in degrees
    :type angle: float
    :return: Angle in degrees normalized to [-180, 180)
    """
    if np.isnan(angle):
        return np.nan
    else:
        while angle >= 180:
            angle -= 2 * 180
        while angle < -180:
            angle += 2 * 180
        return angle


def abs_ang_diff(minuend, subtrahend):
    """
    Returns the smallest difference between two angles
    :param minuend: Angle in [0,2*pi]
    :param subtrahend: Angle in [0,2*pi]
    :return: Angle in [0,2*pi]
    """

    if np.isnan(minuend) or np.isnan(subtrahend):
        return 2*np.pi
    return np.pi - abs(abs(minuend - subtrahend) - np.pi)


def signed_ang_diff(minuend, subtrahend):
    """
    Returns the signed difference between two angles
    :param minuend: Angle in [0,2*pi]
    :param subtrahend: Angle in [0,2*pi]
    :return: Angle in [-2*pi, 2*pi]
    """

    diff = minuend - subtrahend
    diff = (diff + np.pi) % (2*np.pi) - np.pi
    return diff

def getMmsiList(df):
    """
    Returns a list of all different mmsi contained in DataFrame
    :param df: DataFrame containing AIS_data
    :type df: pandas DataFrame
    :return: List of mmsi [mmsi_1, mmsi_2,..., mmsi_n]
    """
    temp_list = df.mmsi.unique().tolist()
    return temp_list


def getListOfMmsiDf(df):
    """
    Returns a list of DataFrames, where each DataFrame contains AIS_data for a ship
    :param df: DataFrame containing AIS_data
    :type df: pandas DataFrame
    :return: List of mmsi [DF_mmsi_1, DF_mmsi_2,..., DF_mmsi_n]
    """
    mmsiList = getMmsiList(df)
    mmsiDfList = [df[df.mmsi == mmsi].reset_index(drop = True) for mmsi in mmsiList]
    return mmsiDfList


def knots_to_mps(knots):
    """
    Transform velocity from knots to m/s
    :type knots: float
    :return: Velocity given in m/s (float)
    """
    if np.isnan(knots) or (knots >= 102.3):
        return np.nan
    mps = knots*1.852/3.6
    return mps


def knots_to_kmph(knots):
    """
    Transform velocity from knots to km/h
    :type knots: float
    :return: Velocity given in km/h (float)
    """
    if np.isnan(knots) or (knots >= 102.3):
        return np.nan
    kmphs = knots*1.852
    return kmphs


def getDisToMeter(lon1, lat1, lon2, lat2, **kwarg):
    """
    Find the distance between two lon/lat - coordinates given in meters
    :type lon1, lat1, lon2, lat2: float
    :return: Distance given in meters (float)
    """
    if lon1 == lon2:
        d_simple = abs(lat1-lat2)*111040
        return round(d_simple, 1)

    if lat1 == lat2:
        d_simple = abs(lon1-lon2)*6362.132*1000*np.pi*2*np.cos(np.deg2rad(lat1))/360
        return round(d_simple, 1)

    if np.isnan(lon1) or np.isnan(lat1) or np.isnan(lon2) or np.isnan(lat2):
        return np.nan
    R = 6362.132
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(a**0.5, (1 - a)**0.5)
    d = R * c * 1000

    return round(d, 1)


def convertSecondsToTime(seconds):
    """
    Convert total seconds to datetime-stamp
    :type seconds: int
    :return: Total time in h:m:s (datetime)
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return datetime.time(hour=hour, minute=minutes, second=seconds)


def calc_trajectory(vessel, index):
    """
    Calculates a projected trajectory of an vessel from an index
    """

    speed = vessel.speed[index]
    lon_start, lat_start = vessel.stateLonLat[:, index]

    course = vessel.true_heading[index - 1]
    if course < 0:
        course += 360
        
    length = vessel.n_msgs - index

    time_passed = 0

    trajectory_lon_lat = np.empty([2, length])

    for i in range(length):
        time_passed += 1
        distance = speed*time_passed*60  # TODO: 1 minute hardcoded

        trajectory_lon_lat[0, i] = lon_start + np.sin(np.deg2rad(course))*distance*360/(6362.132*1000*np.pi*2*np.cos(np.deg2rad(lat_start)))
        trajectory_lon_lat[1, i] = lat_start + np.cos(np.deg2rad(course))*distance/111040

    return trajectory_lon_lat


def calcPredictedCPA(vessel1, vessel2, index):
    ##################
    printer_on = False 
    ##################

    vessel1_trajectory = calc_trajectory(vessel1, index)
    vessel2_trajectory = calc_trajectory(vessel2, index)

    distance = getDisToMeter(vessel1_trajectory[0, 0], vessel1_trajectory[1, 0], vessel2_trajectory[0, 0], vessel2_trajectory[1, 0])
    dist     = distance

    i = 0 
    s = 0
    time_at_cpa = 0

    if printer_on: print("Vessel1: ", vessel1.name)
    if printer_on: print("Vessel2: ", vessel2.name)
    if printer_on: print("Index:", index)

    while i < vessel1.n_msgs - index - 1:
        i += 1
        s += 1

        dist = getDisToMeter(vessel1_trajectory[0, i], vessel1_trajectory[1, i], vessel2_trajectory[0, i], vessel2_trajectory[1, i])

        if np.isnan(dist):
            continue

        if dist > distance:
            break

        if printer_on: print("\t", "*", dist)

        distance = dist
        time_at_cpa = i + index

    if printer_on: print(distance)

    if printer_on:
        import csv
        with open('sums.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(str(s))
    
    return distance, time_at_cpa
