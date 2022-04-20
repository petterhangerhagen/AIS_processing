""" Runs AutoVerify on every case in folders of 'paths'         """
""" All outputs will be saved in /para/ folder                  """
""" Use read.py to merge all csv files in /para/ to one file    """

""" Requirements                                                """
""" -Basemap (used in conjunction with conda and python 3.8.10) """


from AutoVerification import AutoVerification
from AutoVerification import abs_ang_diff
from dataclasses import asdict
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing as mp

csv_file_name = 'rstudio.csv'
save_folder = 'img'

multiple            = True 
cpu_usage           = 80

paths = ['./encs_selected', './encs_north', './encs_south']
#paths = ['./encs_selected', './encs_north', './encs_south']

# ---------------------------------------------------------------------------------------------------
# Combined algorithm
def getCaseParamFromFile(filename, specific_own_name, specific_obst_names, traj, q, row):

    import os.path
    from os import path
        
    if not multiple:
        print(filename)

    ship_path = './' + root.split('/')[1] + '/full_shipdata.csv'
    AV = AutoVerification(AIS_path = path.join(root, filename),
                            ship_path = ship_path,
                            r_colregs_2_max=5000,
                            r_colregs_3_max=3000,
                            r_colregs_4_max=400,
                            r_pref=1500,  # todo: Find the value used by Woerner
                            r_min=1000,
                            r_nm=800,
                            r_col=200,
                            epsilon_course=10,
                            epsilon_speed=2.5,
                            delta_chi_apparent=30,
                            delta_speed_apparent=5,
                            alpha_critical_13=45.0, #45,  # absolute value is used
                            alpha_critical_14=13.0, #13.0,  # absolute value is used
                            alpha_critical_15=-10.0,  # -10.0 
                            alpha_cpa_min_15=-25.0,  # -25
                            alpha_cpa_max_15=165.0,  # 165
                            alpha_ahead_lim_13=45.0,  # absolute value is used
                            phi_OT_min=112.5, # 112.5,  # equal
                            phi_OT_max=247.5) # 247.5,  # equal

    AV.find_ranges() # Find ranges between all ships

    for vessel in AV.vessels:
        
        AV.find_maneuver_detect_index(vessel) # Find maneuvers made by ownship
        for obst in AV.vessels:
            if vessel.id == obst.id:
                continue
            sits = []
            sit_happened = False
            for k in range(AV.n_msgs):
                if AV.entry_criteria(vessel, obst, k) != AV.NAR: # Find applicable COLREG situations between all ships
                    sit_happened = True
                if k == 0:
                    continue
                if AV.situation_matrix[vessel.id, obst.id, k] != AV.situation_matrix[vessel.id, obst.id, k-1]:
                    sits.append(k)
            if sit_happened:
                AV.filterOutNonCompleteSituations(vessel, obst)

    maneuvers = []
    filter_outliners = 2


    start = ' - '
    end = '-60-sec'
    code =  filename[filename.find(start)+len(start):filename.rfind(end)]
    code = code[-5:]

    for vessel in AV.vessels:
        if vessel.travel_dist < 500:
            continue
        plot = False
        for obst in AV.vessels:
            if vessel.id != obst.id and not all(x == 0 for x in AV.situation_matrix[vessel.id, obst.id, :]):
                arr = AV.situation_matrix[vessel.id, obst.id, :]
                arr[0] = AV.NAR
                indxies = np.where(\
                        np.logical_and(arr[:-1] != arr[1:], \
                        np.logical_or(arr[:-1] == 0, arr[1:] == 0)))[0]
                if len(indxies) > 0:
                    plot = True

        if plot:
            if specific_own_name != '' and specific_own_name not in vessel.name:
                continue

            AV.OWN_SHIP = vessel.id

            try:
                q.put((AV.plot_trajectories2(show_trajectory_at = traj, specific_obst_names = specific_obst_names, save = True, save_folder = save_folder), row))
            except Exception as ex:
                print(ex)

#################################################################################
if __name__ == '__main__':

    def saveAndWriteToCSV(name, df, row):
        if 'img_name' not in df:
            df['img_name'] = ""

        df['img_name'][row] = name
        
        return df
    
    def writeToCSV(df):
        df.to_csv(csv_file_name.replace(".csv", "_img.csv"), sep = ';')
        return




    print('STARTING')
    from itertools import chain


    df_read = pd.read_csv(csv_file_name, sep = ';')
    print(df_read)
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    proc = []

    for root, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
        import random
        import sys
        import psutil

        random.shuffle(dirs)
        random.shuffle(files)
        number_of_files = len(files)
        for count, filename in enumerate(files):
            if filename.endswith("60-sec.csv"): 
                
                filename_code = filename.replace("-60-sec.csv","")[-5:]

                #if not filename_code == 'ASEAY':
                #    continue


                df_read_c = df_read[df_read['case'] == filename_code]

                if len(df_read_c) != 0:
                    if not multiple: print(df_read_c)
                else:
                    continue

                for i in range(len(df_read_c)):

                    own_name = df_read_c.own_name.tolist()[i]
                    obst_name = [df_read_c.obst_name.tolist()[i]]
                    traj = df_read_c.maneuver_index_own.tolist()[i]
                    row = df_read_c.index.tolist()[i]

                    if 'img_name' in df_read: 
                        val =  df_read.at[row, 'img_name']
                        if isinstance(val, str):
                            if val != '':
                                continue

                    if not multiple:
                        getCaseParamFromFile(filename, own_name, obst_name, traj, q, row)
                        while not q.empty():
                            n, row = q.get()
                            df_read = saveAndWriteToCSV(n, df_read, row)

                            writeToCSV(df_read)

                        continue

                    p = mp.Process(target = getCaseParamFromFile, args = (filename, own_name, obst_name, traj, q, row,))
                    proc.append(p)
                    p.start()

                    sys.stdout.flush()
                    sys.stdout.write("Working file %s/%s - children working %s. CPU percent %s      \r " % (count, number_of_files, len(proc), psutil.cpu_percent(interval = 0.2)))

                    while psutil.cpu_percent(interval = 0.2) > cpu_usage:
                        sys.stdout.flush()
                        sys.stdout.write("Working file %s/%s - children working %s. CPU percent %s.      \r " % (count, number_of_files, len(proc), psutil.cpu_percent(interval = 0.2)))
                        for ps in proc:
                            ps.join(timeout = 0)
                            if not ps.is_alive():
                                proc.remove(ps)

                                if not q.empty():
                                    n, row = q.get()
                                    df_read = saveAndWriteToCSV(n, df_read, row)

                                    writeToCSV(df_read)
                        time.sleep(0.5)

    sys.stdout.flush()
    print("FINISHING")
    while len(mp.active_children()) > 0:
        for ps in proc:
            ps.join(timeout = 0)
            if not ps.is_alive():
                proc.remove(ps)

                if not q.empty():
                    df_read = saveAndWriteToCSV(n, df_read, row)
                    writeToCSV(df_read)
    while(not q.empty()):
        df_read = saveAndWriteToCSV(n, df_read, row)

    writeToCSV(df_read)

