from pytest import param
from sqlalchemy import case
from AutoVerification import AutoVerification
import os
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime, date, time
from AV_class_functions.helper_methods import knots_to_mps

""" Runs AutoVerify on every case in folders of 'paths'    
All outputs will be saved in /para/ folder                  
Use read.py to merge all csv files in /para/ to one file   
Requirements:                                                
-Basemap (used in conjunction with conda and python 3.8.10) """

csv_file_name = './testing/Params - 01-02-2019, 12-28-11 - 6MC0Q.csv'
save_folder = 'img'

multiple = False
cpu_usage = 80

#paths = ['./encs_selected', './encs_north', './encs_south']s
paths = ['./encs_west_selected']


# ---------------------------------------------------------------------------------------------------
# Combined algorithm
def get_case_param_from_file(filename, specific_own_name, specific_obst_names, man_idx, q, df_row):
    if not multiple:
        print(filename)

    ship_path = './' + root.split('/')[1] + '/full_shipdata.csv'
    av = AutoVerification(ais_path=os.path.join(root, filename),
                          ship_path=ship_path,
                          r_colregs_2_max=5000,
                          r_colregs_3_max=3000,
                          r_colregs_4_max=400,
                          epsilon_course=10,
                          epsilon_speed=2.5,
                          alpha_critical_13=45.0,
                          alpha_critical_14=13.0,
                          alpha_critical_15=-10.0,
                          phi_OT_min=112.5,
                          phi_OT_max=247.5)

    av.find_ranges()  # Find ranges between all ships

    for vessel in av.vessels:
        av.find_maneuver_detect_index(vessel)  # Find maneuvers made by ownship

        for obst in av.vessels:
            if vessel.id == obst.id:
                continue
            sits = []
            sit_happened = False
            for k in range(av.n_msgs):
                if av.entry_criteria(vessel, obst, k) != av.NAR:  # Find applicable COLREG rules between all ships
                    sit_happened = True
                if k == 0:
                    continue
                if av.situation_matrix[vessel.id, obst.id, k] != av.situation_matrix[vessel.id, obst.id, k - 1]:
                    sits.append(k)
            if sit_happened:
                av.filterOutNonCompleteSituations(vessel, obst)

    for vessel in av.vessels:
        if vessel.travel_dist < 500:
            continue
        plot = False
        for obst in av.vessels:
            if vessel.id != obst.id and not all(x == 0 for x in av.situation_matrix[vessel.id, obst.id, :]):
                arr = av.situation_matrix[vessel.id, obst.id, :]
                arr[0] = av.NAR
                indices = np.where(np.logical_and(arr[:-1] != arr[1:], np.logical_or(arr[:-1] == 0, arr[1:] == 0)))[0]
                if len(indices) > 0:
                    plot = True

        if plot:
            if specific_own_name != '' and specific_own_name not in vessel.name:
                continue

            av.OWN_SHIP = vessel.id
            try:
                q.put((av.plot_trajectories(show_trajectory_at=man_idx, specific_obst_names=specific_obst_names,
                                            save=True, save_folder=save_folder), df_row))
            except Exception as ex:
                print(ex)


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
    # TODO: Plot cpa when the index has been added to the parameter file
    # TODO: Plot trajectory prediction at maneuver stop

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
    sit_start = para_df['start_idx'] 
    sit_stop = para_df['stop_idx'] 
    

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

    # Plot trajectories
    mapping.plot(os_x, os_y, color='b', label='ownship')
    mapping.plot(tg_x, tg_y, color='r', label='obstacle')

    # Mark significant indices
    mapping.scatter(os_x[0], os_y[0], color='b', marker='x', label='start')
    mapping.scatter(tg_x[0], tg_y[0], color='r', marker='x')

    mapping.scatter(os_x[sit_start], os_y[sit_start], color='g', marker='d')
    mapping.scatter(tg_x[sit_start], tg_y[sit_start], color='g', marker='d')
    ax.annotate('COLREG\nstart', (os_x[sit_start], os_y[sit_start]), xytext=(os_x[sit_start] + 5, os_y[sit_start] + 5))
    mapping.scatter(os_x[sit_stop], os_y[sit_stop], color='y', marker='d')
    mapping.scatter(tg_x[sit_stop], tg_y[sit_stop], color='y', marker='d')
    ax.annotate('COLREG\nstop', (os_x[sit_stop], os_y[sit_stop]), xytext=(os_x[sit_stop] + 5, os_y[sit_stop] + 5))

    
    if para_df['maneuver_made_own']: 

        # Plot predicted trajectory at index before maneuver
        if isinstance(para_df['time'], str): 
            time_delta = datetime.strptime(para_df['time'], "%H:%M:%S") - datetime(1900, 1, 1) 
        else:  # Assuming datetime.time
            time_delta = datetime.combine(date.min, para_df['time']) - datetime.min  # if 'time' formatted as 
        dt = time_delta.total_seconds() / (para_df['stop_idx'] - para_df['start_idx'])
        n_msgs = len(os_x)
        speed = knots_to_mps(os_df['sog'].values[maneuver_start - 1])
        course = os_df['cog'].values[maneuver_start - 1]
        pred_traj = predict_trajectory(os_df['lon'].values, os_df['lat'].values, maneuver_start - 1,
                                       n_msgs, dt, speed, course)
        pred_x, pred_y = mapping(pred_traj[0, :], pred_traj[1, :])
        mapping.plot(pred_x, pred_y, color='k', linestyle=':', label='prediction')

        mapping.scatter(os_x[maneuver_start], os_y[maneuver_start], color='b', marker='o', label='maneuver')
        title_string = ''
    else:
        title_string = '\n no evasive maneuver'
    ax.legend()
    title_string = str(para_df['own_name']) + ' - ' + str(para_df['obst_name']) + title_string 
    ax.set_title(title_string)

    fig.set_size_inches(10.4, 8.8)

    img_name = './' + str(save_folder) + '/' + str(para_df['own_name']) + '-' + str(para_df['obst_name']) + '-' + str(para_df['case']) + '.png'
    plt.savefig(img_name)

    #plt.show()
    
    return img_name


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

                # filename_code = file_name.replace("-60-sec.csv", "")[-5:]  # filename must be ais data
                start = ' - '
                end = '-60-sec'
                filename_code = file_name[file_name.find(start) + len(start):file_name.rfind(end)][-5:]
                param_df = df_read[df_read['case'] == filename_code]  # Parameter data

                # # If it doesn't work to use the case_path variable, try this:
                # for ais_root, ais_directory, ais_files in os.walk(root):
                #     for ais_file in ais_files:
                #         if ais_file.endswith(".csv") and (filename_code + '-60-sec' in ais_file):
                #             case_path = os.path.join(ais_root, ais_file)
                #             break
                case_path = os.path.join(root, file_name)
                ais_df = pd.read_csv(case_path, sep=';')  # AIS data

                # TODO: Write plotting functions to repplace get_case_param_from_file and av.plot_trajectories to avoid
                #  creation of Autoverification instance
                if len(param_df) != 0:
                    if not multiple:
                        print(param_df)
                else:
                    continue

                for i in range(len(param_df)):
                    own_mmsi = param_df.own_mmsi.tolist()[i]             
                    obst_mmsi = param_df.obst_mmsi.tolist()[i]      
                    maneuver_idx = param_df.maneuver_index_own.tolist()[i]
                    row = param_df.index.tolist()[i]

                    ownship_df = ais_df.loc[ais_df['mmsi'] == own_mmsi] 
                    obst_df = ais_df.loc[ais_df['mmsi'] == obst_mmsi]   
                    sit_df = param_df.iloc[row] 

                    if 'img_name' in df_read:
                        val = df_read.at[row, 'img_name']
                        if isinstance(val, str):
                            if val != '':
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
                        time.sleep(0.5)

    sys.stdout.flush()
    print("FINISHING")
    while len(mp.active_children()) > 0:
        for ps in proc:
            ps.join(timeout=0)
            if not ps.is_alive():
                proc.remove(ps)

                if not queue.empty():
                    df_read = save2dataframe(img_name, df_read, row)
                    write2csv(df_read)
    while not queue.empty():
        df_read = save2dataframe(img_name, df_read, row)

    write2csv(df_read)
