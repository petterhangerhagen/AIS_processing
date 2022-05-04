from pytest import param
from sqlalchemy import case
from AutoVerification import AutoVerification
import os
import time as t
import pandas as pd
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime, date, time
from AV_class_functions.helper_methods import knots_to_mps
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

""" Runs AutoVerify on every case in folders of 'paths'    
All outputs will be saved in /para/ folder                  
Use read.py to merge all csv files in /para/ to one file   
Requirements:                                                
-Basemap (used in conjunction with conda and python 3.8.10) """

csv_file_name = 'superPara.csv'
save_folder = 'img'

multiple = True
cpu_usage = 80

# paths = ['./encs_selected', './encs_north', './encs_south']s
paths = ['./encs']


# ---------------------------------------------------------------------------------------------------
def get_params_and_plot(para_df, os_df, tg_df, q, df_row):
    try:
        q.put((plot_situation(para_df, os_df, tg_df), df_row))
    except Exception as ex:
        print(ex)


def plot_situation(para_df, os_df, tg_df):
    """Plot single situation from dataframes containing AIS data and extracted parameters.
    The plots include:
    * start and stop of COLREGS situation
    * start point of the maneuver registered in the parameter file
    * cpa
    :param para_df: Dataframe containing parameters extracted from the situation
    :param os_df: Dataframe containing AIS data from vessel considered the ownship
    :param tg_df: Dataframe containing AIS data from vessel considered the obstacle"""

    x_min = 9.0 * 10 ** 9
    x_max = -9.0 * 10 ** 9
    y_min = x_min
    y_max = x_max
    x_min_temp = min(min(os_df['lon'].values), min(tg_df['lon'].values))
    x_max_temp = max(max(os_df['lon'].values), max(tg_df['lon'].values))
    x_min = min(x_min, x_min_temp)
    x_max = max(x_max, x_max_temp)
    y_min_temp = min(min(os_df['lat'].values), min(tg_df['lat'].values))
    y_max_temp = max(max(os_df['lat'].values), max(tg_df['lat'].values))
    y_min = min(y_min, y_min_temp)
    y_max = max(y_max, y_max_temp)
    margin = 0.02
    lon_bounds = [x_min - margin, x_max + margin]
    lat_bounds = [y_min - margin, y_max + margin]

    maneuver_start = int(para_df['maneuver_index_own'])
    if para_df['maneuver_stop_idx_own'] is None:
        maneuver_stop = None
    else:
        maneuver_stop = int(para_df['maneuver_stop_idx_own'])
    sit_start = para_df['start_idx']
    sit_stop = para_df['stop_idx']
    cpa_idx = para_df['cpa_idx']

    fig, ax = plt.subplots()
    # Prepare mapping
    mapping = Basemap(projection='merc',
                      lat_0=lat_bounds[0], lon_0=lon_bounds[0],
                      llcrnrlat=lat_bounds[0], urcrnrlat=lat_bounds[1],
                      llcrnrlon=lon_bounds[0], urcrnrlon=lon_bounds[1], resolution='h', ax=ax)
    mapping.drawcoastlines(linewidth=0.25)
    mapping.fillcontinents(color='coral', lake_color='aqua')

    # Map trajectories to chosen projection
    os_x, os_y = mapping(os_df['lon'].values, os_df['lat'].values)
    tg_x, tg_y = mapping(tg_df['lon'].values, tg_df['lat'].values)

    # Plot trajectories and situation
    sit_len = sit_stop - sit_start
    os_ln, = mapping.plot(os_x[:sit_start + 1], os_y[:sit_start + 1], c='b', ls='--')
    tg_ln, = mapping.plot(tg_x[:sit_start + 1], tg_y[:sit_start + 1], c='r', ls='--')
    sit_ln, = mapping.plot(os_x[sit_start:sit_stop + 1], os_y[sit_start:sit_stop + 1],
                           marker='1', markevery=sit_len, c='b')
    mapping.plot(tg_x[sit_start:sit_stop + 1], tg_y[sit_start:sit_stop + 1], marker='1', markevery=sit_len, c='r')
    mapping.plot(os_x[sit_stop:], os_y[sit_stop:], c='b', ls='--')
    mapping.plot(tg_x[sit_stop:], tg_y[sit_stop:], c='r', ls='--')

    # Mark significant indices
    os_start_mrk = mapping.scatter(os_x[0], os_y[0], c='b', marker='X', s=60)
    tg_start_mrk = mapping.scatter(tg_x[0], tg_y[0], c='r', marker='X', s=60)
    os_cpa_mrk = mapping.scatter(os_x[cpa_idx], os_y[cpa_idx], c='b', marker='x', s=100)
    tg_cpa_mrk = mapping.scatter(tg_x[cpa_idx], tg_y[cpa_idx], c='r', marker='x', s=100)

    handles = [os_ln, tg_ln, sit_ln, (os_start_mrk, tg_start_mrk), (os_cpa_mrk, tg_cpa_mrk)]
    labels = ['ownship', 'obstacle', 'situation', 'start points', 'cpa']

    if para_df['maneuver_made_own']:
        # Plot predicted trajectory at index before maneuver
        if isinstance(para_df['time'], str): 
            time_delta = datetime.strptime(para_df['time'], "%H:%M:%S") - datetime(1900, 1, 1) 
        else:  # Assuming datetime.time
            time_delta = datetime.combine(date.min, para_df['time']) - datetime.min
        dt = time_delta.total_seconds() / (para_df['stop_idx'] - para_df['start_idx'])

        n_msgs = len(os_x)
        speed = knots_to_mps(os_df['sog'].values[maneuver_start - 1])
        course = os_df['cog'].values[maneuver_start - 1]
        pred_traj = predict_trajectory(os_df['lon'].values, os_df['lat'].values, maneuver_start - 1,
                                       n_msgs, dt, speed, course)
        pred_x, pred_y = mapping(pred_traj[0, :], pred_traj[1, :])
        pred_ln, = mapping.plot(pred_x, pred_y, color='k', linestyle=':')

        man_start_mrk = mapping.scatter(os_x[maneuver_start], os_y[maneuver_start], c='seagreen', marker='o', s=80)
        obst_at_man_mrk = mapping.scatter(tg_x[maneuver_start], tg_y[maneuver_start], edgecolors='seagreen', marker='o',
                                          facecolors='none', s=80)
        if maneuver_stop is not None:
            man_stop_mrk = mapping.scatter(os_x[maneuver_stop], os_y[maneuver_stop], c='darkorange', marker='o', s=80)
            handles.append((man_start_mrk, man_stop_mrk))
            handles.append(obst_at_man_mrk)
            labels.append('maneuver start/stop')
            labels.append('obst pos at os man')
        else:
            handles.append(man_start_mrk)
            handles.append(obst_at_man_mrk)
            labels.append('maneuver start')
            labels.append('obst pos at os man')

        handles.append(pred_ln)
        handles.append('pre manuver pred')
        title_string = ''
    else:
        title_string = '\n no evasive maneuver'

    handler_map = {sit_ln: HandlerLine2D(numpoints=2), tuple: HandlerTuple(ndivide=None)}

    ax.legend(handles, labels, scatterpoints=1, handler_map=handler_map)

    title_string = str(para_df['own_name']) + ' - ' + str(para_df['obst_name']) + title_string

    ax.set_title(title_string)

    fig.set_size_inches(10.4, 8.8)

    image_name = './' + str(save_folder) + '/' + str(para_df['own_mmsi']) + '-' + str(para_df['obst_mmsi']) + '-' \
                 + str(para_df['case']) + '_' + str(para_df['COLREG']) + '.png'
    plt.savefig(image_name)
    
    return image_name


def predict_trajectory(lon, lat, index, n_msgs, dt, speed, course):
    """
    Predict a straight trajectory for a vessel at given index.
    :param lon: Longitudinal positions of original trajectory samples.
    :param lat: Latitudinal positions of original trajectory samples.
    :param index: Sample index where to start prediction.
    :param n_msgs: Number of samples in original trajectory.
    :param dt: Sample interval.
    :param speed: Speed of vessel at index in meters per second
    :param course: Course angle at index in degrees.
    """

    curr_pos = np.array([lon[index], lat[index]])

    if course < 0:
        course += 360

    n_steps = n_msgs - index + 1
    step = 0
    trajectory = np.empty([2, n_steps])
    trajectory[:, 0] = curr_pos

    for k in range(1, n_steps):
        step += 1
        distance = speed * step * dt

        trajectory[0, k] = lon[index] + np.sin(np.deg2rad(course)) * distance * 360 / (
                    6362.132 * 1000 * np.pi * 2 * np.cos(np.deg2rad(lat[index])))
        trajectory[1, k] = lat[index] + np.cos(np.deg2rad(course)) * distance / 111040

    return trajectory


#################################################################################


if __name__ == '__main__':

    def save2dataframe(name, df, df_row):
        if 'img_name' not in df:
            df['img_name'] = ""
        df['img_name'][df_row] = name
        return df


    def write2csv(df):
        df.to_csv(csv_file_name.replace(".csv", "_img.csv"), sep=';')
        return


    print('STARTING')
    from itertools import chain

    df_read = pd.read_csv(csv_file_name, sep=';')
    print(df_read)
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    proc = []

    for root, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
        import random
        import sys
        import psutil

        random.shuffle(dirs)  # Why is this shuffle necessary?
        random.shuffle(files)  # Why is this shuffle necessary?
        number_of_files = len(files)
        for count, file_name in enumerate(files):
            if file_name.endswith("60-sec.csv"):

                start = ' - '
                end = '-60-sec'
                filename_code = file_name[file_name.find(start) + len(start):file_name.rfind(end)][-5:]
                param_df = df_read[df_read['case'] == filename_code]  # Parameter data

                case_path = os.path.join(root, file_name)
                ais_df = pd.read_csv(case_path, sep=';')  # AIS data

                if len(param_df) != 0:
                    if not multiple:
                        print(param_df)
                else:
                    continue

                for i in range(len(param_df)):
                    own_mmsi = param_df.own_mmsi.tolist()[i]             
                    obst_mmsi = param_df.obst_mmsi.tolist()[i]      
                    maneuver_idx = param_df.maneuver_index_own.tolist()[i]
                    maneuver_made = param_df.maneuver_made_own.tolist()[i]
                    row = param_df.index.tolist()[i]

                    ownship_df = ais_df.loc[ais_df['mmsi'] == own_mmsi] 
                    obst_df = ais_df.loc[ais_df['mmsi'] == obst_mmsi]   
                    sit_df = param_df.iloc[i] 

                    if 'img_name' in df_read:
                        val = df_read.at[row, 'img_name']
                        if isinstance(val, str):
                            if val != '':
                                continue
                    if not maneuver_made:
                        continue

                    if not multiple:
                        # get_case_param_from_file(file_name, own_name, obst_name, maneuver_idx, queue, row)
                        get_params_and_plot(sit_df, ownship_df, obst_df, queue, row)
                        while not queue.empty():
                            img_name, row = queue.get()
                            df_read = save2dataframe(img_name, df_read, row)

                            write2csv(df_read)

                        continue

                    # p = mp.Process(target=get_case_param_from_file,
                    #                args=(file_name, own_name, obst_name, maneuver_idx, queue, row,))
                    p = mp.Process(target=get_params_and_plot,
                                   args=(sit_df, ownship_df, obst_df, queue, row,))
                    proc.append(p)
                    p.start()

                    sys.stdout.flush()
                    sys.stdout.write("Working file %s/%s - children working %s. CPU percent %s      \r " % (
                        count, number_of_files, len(proc), psutil.cpu_percent(interval=0.2)))

                    while psutil.cpu_percent(interval=0.2) > cpu_usage:
                        sys.stdout.flush()
                        sys.stdout.write("Working file %s/%s - children working %s. CPU percent %s.      \r " % (
                            count, number_of_files, len(proc), psutil.cpu_percent(interval=0.2)))
                        for ps in proc:
                            ps.join(timeout=0)
                            if not ps.is_alive():
                                proc.remove(ps)

                                if not queue.empty():
                                    img_name, row = queue.get()
                                    df_read = save2dataframe(img_name, df_read, row)

                                    write2csv(df_read)
                        t.sleep(0.5)

    sys.stdout.flush()
    print("FINISHING")
    while len(mp.active_children()) > 0:
        for ps in proc:
            ps.join(timeout=0)
            if not ps.is_alive():
                proc.remove(ps)

                if not queue.empty():
                    img_name, row = queue.get()
                    df_read = save2dataframe(img_name, df_read, row)
                    write2csv(df_read)
    while not queue.empty():
        df_read = save2dataframe(img_name, df_read, row)

    write2csv(df_read)
