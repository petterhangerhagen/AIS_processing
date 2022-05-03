from AutoVerification import AutoVerification
from dataclasses import asdict
import os
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
from itertools import chain

""" Extracts parameters from the cases
Can also be used to plot cases/situations
Plotting requires basemap (used with conda and python 3.8.10)   """

multiple = True

plot_trajectories = False
plot_case = False

SPECIFIC_OWN_NAME = ''  # Set to '' for no specific and automatic plotting

SPECIFIC_OBST_NAMES = ['']  # Set to '' for no specific and automatic plotting

unique_case = ''  # Set to '' for non unique case-code (plot all).
# To plot specific case add case-code of the file

TRAJ = None  # Set to None if automatic trajectory detection

cpu_usage = 80
overwrite = False

paths = ['./encs_selected', './encs_north', './encs_south']


# ---------------------------------------------------------------------------------------------------
# Combined algorithm
def get_case_param_from_file(filename):
    if not multiple and unique_case != '' and unique_case not in filename:
        return

    from os import path

    if not multiple:
        print(path.join(root, filename))

    if overwrite:
        start = ' - '
        end = '-60-sec'
        code = filename[filename.find(start) + len(start):filename.rfind(end)]
        param_filename = './para/Params - ' + code + '.csv'
        import os.path
        from os import path
        if path.exists(param_filename):
            return

    ship_path = './' + root.split('/')[1] + '/''full_shipdata.csv'
    AV = AutoVerification(ais_path=path.join(root, filename),
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

    AV.find_ranges()  # Find ranges between all ships

    for vessel in AV.vessels:

        AV.find_maneuver_detect_index(vessel)  # Find maneuvers made by ownship
        for obst in AV.vessels:
            if vessel.id == obst.id:
                continue
            sits = []
            sit_happened = False
            for k in range(AV.n_msgs):
                if AV.entry_criteria(vessel, obst, k) != AV.NAR:  # Find applicable COLREG rules between all ships
                    sit_happened = True
                if k == 0:
                    continue
                if AV.situation_matrix[vessel.id, obst.id, k] != AV.situation_matrix[vessel.id, obst.id, k - 1]:
                    sits.append(k)
            if sit_happened:
                AV.filter_out_non_complete_situations(vessel, obst)

    start = ' - '
    end = '-60-sec'
    code = filename[filename.find(start) + len(start):filename.rfind(end)]
    code = code[-5:]

    params = []
    for vessel in AV.vessels:
        if vessel.travel_dist < 500 or 7 in vessel.nav_status:
            continue
        plot = False

        # Get vessel name
        if SPECIFIC_OWN_NAME == '':
            specific_own_name = vessel.name
        else:
            specific_own_name = SPECIFIC_OWN_NAME
        specific_obst_names = SPECIFIC_OBST_NAMES

        for obst in AV.vessels:
            if vessel.id != obst.id and not all(x == 0 for x in AV.situation_matrix[vessel.id, obst.id, :]):
                arr = AV.situation_matrix[vessel.id, obst.id, :]
                arr[0] = AV.NAR
                end_pt_indices = np.where(np.logical_and(arr[:-1] != arr[1:],
                                                         np.logical_or(arr[:-1] == 0, arr[1:] == 0)))[0]

                if len(end_pt_indices) > 0:
                    plot = True

                for start, stop in zip(end_pt_indices[0::2], end_pt_indices[1::2]):
                    if stop - start < 5:
                        continue
                    sit_params = AV.constructParams(vessel, obst, start + 1, stop + 1)

                    if sit_params is None:
                        continue

                    param_dict = asdict(sit_params)

                    if False and not multiple:
                        print(param_dict)  # debug disabled

                    if SPECIFIC_OBST_NAMES == ['']:
                        specific_obst_names = [obst.name]

                    param_dict['case'] = code
                    param_dict['dataset'] = root.split('/')[1]
                    params.append(param_dict)

                    if plot and plot_trajectories and not multiple:
                        if specific_obst_names == ['']:
                            specific_obst_names = [obst.name]

                        if vessel.name == specific_own_name:
                            AV.OWN_SHIP = vessel.id
                            if TRAJ is None:
                                traj = param_dict['maneuver_index_own']
                            else:
                                traj = TRAJ
                            AV.plot_trajectories(show_trajectory_at=traj,
                                                 specific_obst_names=specific_obst_names)

    if plot_case:
        AV.plot_case()

    data_bank = pd.DataFrame(params)
    start = ' - '
    end = '-60-sec'
    code = filename[filename.find(start) + len(start):filename.rfind(end)]
    data_bank.to_csv('./para/Params - ' + code + '.csv', sep=';', index=False)
    del data_bank


#################################################################################
print('STARTING')


for root, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
    proc = []
    import random
    import sys
    import psutil

    random.shuffle(dirs)
    random.shuffle(files)
    number_of_files = len(files)

    for count, file_name in enumerate(files):
        if file_name.endswith("60-sec.csv"):
            if not multiple:
                get_case_param_from_file(file_name)
                continue
            p = mp.Process(target=get_case_param_from_file, args=(file_name,))
            proc.append(p)
            p.start()

            sys.stdout.flush()
            sys.stdout.write(
                "Working file %s/%s - children working %s        \r " % (count, number_of_files, len(proc)))

            while psutil.cpu_percent() > cpu_usage:
                for ps in proc:
                    ps.join(timeout=0)
                    if not ps.is_alive():
                        proc.remove(ps)
                time.sleep(0.5)

    while len(mp.active_children()) > 0:
        time.sleep(0.5)
