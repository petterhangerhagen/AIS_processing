from AutoVerification import AutoVerification
import os
import time
import pandas as pd
import numpy as np
import multiprocessing as mp

""" Runs AutoVerify on every case in folders of 'paths'    
All outputs will be saved in /para/ folder                  
Use read.py to merge all csv files in /para/ to one file   
Requirements:                                                
-Basemap (used in conjunction with conda and python 3.8.10) """

csv_file_name = 'rstudio.csv'
save_folder = 'img'

multiple = True
cpu_usage = 80

paths = ['./encs_selected', './encs_north', './encs_south']


# ---------------------------------------------------------------------------------------------------
# Combined algorithm
def get_case_param_from_file(filename, specific_own_name, specific_obst_names, man_idx, q, df_row):

    from os import path
        
    if not multiple:
        print(filename)

    ship_path = './' + root.split('/')[1] + '/full_shipdata.csv'
    av = AutoVerification(AIS_path=path.join(root, filename),
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
                if av.situation_matrix[vessel.id, obst.id, k] != av.situation_matrix[vessel.id, obst.id, k-1]:
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
                q.put((av.plot_trajectories2(show_trajectory_at=man_idx, specific_obst_names=specific_obst_names,
                                             save=True, save_folder=save_folder), df_row))
            except Exception as ex:
                print(ex)


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

        random.shuffle(dirs)
        random.shuffle(files)
        number_of_files = len(files)
        for count, file_name in enumerate(files):
            if file_name.endswith("60-sec.csv"):
                
                filename_code = file_name.replace("-60-sec.csv", "")[-5:]
                df_read_c = df_read[df_read['case'] == filename_code]

                if len(df_read_c) != 0:
                    if not multiple:
                        print(df_read_c)
                else:
                    continue

                for i in range(len(df_read_c)):
                    own_name = df_read_c.own_name.tolist()[i]
                    obst_name = [df_read_c.obst_name.tolist()[i]]
                    maneuver_idx = df_read_c.maneuver_index_own.tolist()[i]
                    row = df_read_c.index.tolist()[i]

                    if 'img_name' in df_read: 
                        val = df_read.at[row, 'img_name']
                        if isinstance(val, str):
                            if val != '':
                                continue

                    if not multiple:
                        get_case_param_from_file(file_name, own_name, obst_name, maneuver_idx, queue, row)
                        while not queue.empty():
                            img_name, row = queue.get()
                            df_read = save2dataframe(img_name, df_read, row)

                            write2csv(df_read)

                        continue

                    p = mp.Process(target=get_case_param_from_file,
                                   args=(file_name, own_name, obst_name, maneuver_idx, queue, row,))
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

