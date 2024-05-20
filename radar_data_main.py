import numpy as np
from AutoVerification import AutoVerification, Vessel
from scipy.ndimage.filters import gaussian_filter
import plotting
import sys
import matplotlib.pyplot as plt
import os
from video import Video

def read_out_radar_data(data_file):
    # The message need to be of this form [x y yaw u v]
    vessels = []
    data = np.load(data_file, allow_pickle=True).item()

    timestamps_dict = {}

    for vessel_id, track in data.items():
        # if vessel_id == 1:
        #     continue
        timestamps, x_positions, y_positions, yaws, x_velocities, y_velocities = zip(*track)
        timestamps_dict[vessel_id] = timestamps

    total_timestamps = set()
    for vessel_id, timestamps in timestamps_dict.items():
        total_timestamps.update(timestamps)
    total_timestamps = sorted(list(total_timestamps))

    for vessel_id, track in data.items():
        # if vessel_id == 1:
        #     continue
        timestamps, x_positions, y_positions, yaws, x_velocities, y_velocities = zip(*track)
        timestamps = list(timestamps)

        new_timestamps = np.zeros(len(total_timestamps))
        j = 0
        for i in range(len(total_timestamps)):
            if total_timestamps[i] == timestamps[j]:
                new_timestamps[i] = timestamps[j]
                # j can not be greater than the length of timestamps
                if j < len(timestamps) - 1:
                    j += 1
            else:
                new_timestamps[i] = np.nan

        x_positions_new = np.zeros(len(total_timestamps))
        y_positions_new = np.zeros(len(total_timestamps))
        yaws_new = np.zeros(len(total_timestamps))
        x_velocities_new = np.zeros(len(total_timestamps))
        y_velocities_new = np.zeros(len(total_timestamps))

        for i in range(len(new_timestamps)):
            if np.isnan(new_timestamps[i]):
                x_positions_new[i] = np.nan
                y_positions_new[i] = np.nan
                yaws_new[i] = np.nan
                x_velocities_new[i] = np.nan
                y_velocities_new[i] = np.nan
            else:
                index = timestamps.index(new_timestamps[i])
                x_positions_new[i] = x_positions[index]
                y_positions_new[i] = y_positions[index]
                yaws_new[i] = yaws[index]
                x_velocities_new[i] = x_velocities[index]
                y_velocities_new[i] = y_velocities[index]
                
        ### Create vessel object
        vessel = Vessel(vessel_id, len(total_timestamps))
        vessel.time_stamps = total_timestamps
        vessel.id = vessel_id
        vessel.stateDateTime = total_timestamps
        for j in range(len(total_timestamps)):
            vessel.state[0, j] = x_positions_new[j]
            vessel.state[1, j] = y_positions_new[j]
            vessel.state[2, j] = np.deg2rad(yaws_new[j])
            vessel.state[3, j] = x_velocities_new[j]
            vessel.state[4, j] = y_velocities_new[j]

        # Logic for finding the first and last nan index, the nan index is the index of the first and last non nan value
        first_element_nan = False
        for k, elem in enumerate(vessel.state[0]):
            if k == 0:
                if np.isnan(elem):
                    first_element_nan = True
                else:
                    vessel.nan_idx[0] = k
                    break

            if first_element_nan:
                if not np.isnan(elem):
                    vessel.nan_idx[0] = k
                    break
        for k in range(vessel.nan_idx[0][0], len(vessel.state[0])):
            if np.isnan(vessel.state[0, k]):
                vessel.nan_idx[1] = k -1 
                break
            if k == len(total_timestamps) - 1:
                vessel.nan_idx[1] = k

        sog = np.sqrt(np.square(vessel.state[3]) + np.square(vessel.state[4]))
        vessel.speed = sog

        dt = np.diff(total_timestamps)
        dt = np.append(dt, dt[-1])
        dt = np.mean(dt)
        vessel.dT = dt

        vessel.travel_dist = np.linalg.norm(
                [vessel.state[0, vessel.nan_idx[1]] - vessel.state[0, vessel.nan_idx[0]], \
                 vessel.state[1, vessel.nan_idx[1]] - vessel.state[1, vessel.nan_idx[0]]])
     
        if vessel.travel_dist > 50:

                # Calculate derivative of speed
                speed = np.array(vessel.speed)

                target_area = np.isnan(speed) == False
                speed[target_area] = gaussian_filter(speed[target_area], sigma=1)

                target_area = [np.logical_and(np.logical_and(target_area[i] == True, target_area[i + 2] == True),
                                              target_area[i + 1] == True) for i in range(len(target_area) - 2)]
                target_area = np.append(False, target_area)
                target_area = np.append(target_area, False)
                if speed.size >= 3:
                    vessel.speed_der[:] = [0 for i in range(len(vessel.speed))]
                    speed = speed[np.isnan(speed) == False]
                    try:
                        vessel.speed_der[target_area] = [np.dot([speed[i], speed[i + 1], speed[i + 2]], [-0.5, 0, 0.5])
                                                         for i in range(len(speed) - 2)]
                    except:
                        pass

                        # Calculate derivatives of yaw
                a = np.array(vessel.state[2, :])
                d = np.append([0], a[1:] - a[:-1], 0)

                d[np.isnan(d)] = 0
                d[abs(d) < np.pi] = 0
                d[d < -np.pi] = -2 * np.pi
                d[d > np.pi] = 2 * np.pi  # d is now 2pi or -2pi at jumps from pi to -pi or opposite

                s = np.cumsum(d, axis=0)  # sum of all previuos changes

                target_area = np.isnan(a) == False

                a[target_area] = a[target_area] - s[
                    target_area]  # this is to not have sudden changes from pi to -pi or opposite count as maneuvers

                a[target_area] = gaussian_filter(a[target_area], sigma=2)

                target_area = [np.logical_and(target_area[i] == True, True == target_area[i + 2]) for i in
                               range(len(target_area) - 2)]
                target_area = np.append(False, target_area)
                target_area = np.append(target_area, False)
                if a.size >= 3:
                    a = a[np.isnan(a) == False]
                    vessel.maneuver_der[0, :] = [0 for i in range(len(vessel.state[2, :]))]
                    vessel.maneuver_der[0, target_area] = [np.dot([a[i], a[i + 1], a[i + 2]], [-0.5, 0, 0.5]) for i in
                                                           range(len(a) - 2)]
                    vessel.maneuver_der[1, :] = [0 for i in range(len(vessel.state[2, :]))]
                    vessel.maneuver_der[1, target_area] = [np.dot([a[i], a[i + 1], a[i + 2]], [1, -2, 1]) for i in
                                                           range(len(a) - 2)]

                    target_area = [np.logical_and(target_area[i] == True, True == target_area[i + 2]) for i in
                                   range(len(target_area) - 2)]
                    target_area = np.append(False, target_area)
                    target_area = np.append(target_area, False)
                    vessel.maneuver_der[2, :] = [0 for i in range(len(vessel.state[2, :]))]
                    vessel.maneuver_der[2, target_area] = [
                        np.dot([a[i], a[i + 1], a[i + 2], a[i + 3], a[i + 4]], [-0.5, 1, 0, -1, 0.5]) for i in
                        range(len(a) - 4)]
                    vessel.maneuver_der[1, :] = [0 for i in range(len(vessel.state[2, :]))]
                    vessel.maneuver_der[1, target_area] = [
                        np.dot([a[i], a[i + 1], a[i + 2], a[i + 3], a[i + 4]], [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
                        for i in range(len(a) - 4)]
                    
        if vessel.travel_dist > 20:
            vessels.append(vessel)
        for id_idx, vessel in enumerate(vessels):
            vessel.id = id_idx
    return vessels

def all_elements_zero_or_OP(matrix):
    for row in matrix:
        for element in row:
            if element != 0 and element != -3:  # Modified condition
                return False
    return True

def write_scenario_to_file(data_file):
    data_file = data_file.split("/")[-1]

    txt_file = "scenarios_with_colreg_situations.txt"
    with open(txt_file, 'r') as f:
        # need to check if the name is already written
        lines = f.readlines()
        already_written = False
        for line in lines:
            if data_file == line[:-1]:
                already_written = True
                break
    with open(txt_file, 'a') as f:
        if not already_written:
            f.write(data_file + "\n")

def list_npy_files():
    colreg_files_dir = "/home/aflaptop/Documents/radar_tracker/Radar-data-processing-and-analysis/code/colreg_files"
    directory = colreg_files_dir
    npy_files = [file for file in os.listdir(directory) if file.endswith('.npy')]
    new_files = []
    for file in npy_files:
        new_files.append(os.path.join(directory, file))
    return new_files

def list_local_npy_files():
    colreg_files_dir = "/home/aflaptop/Documents/Scripts/AIS_processing/npy_files"
    directory = colreg_files_dir
    npy_files = [file for file in os.listdir(directory) if file.startswith('colreg_tracks_rosbag') and file.endswith('.npy')]
    new_files = []
    for file in npy_files:
        new_files.append(os.path.join(directory, file))
    return new_files

def find_only_colreg_files(npy_files):
    path_list = []
    colreg_situation_files = []
    with open("scenarios_with_colreg_situations.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            colreg_situation_files.append(line[:-1])
    temp_npy_files = []
    for file in npy_files:
        temp_npy_files.append(file.split("/")[-1])
    
    for file in temp_npy_files:
        for colreg_file in colreg_situation_files:
            if file == colreg_file:
                path_list.append(file)
    
    new_path_list = []
    colreg_files_dir = "/home/aflaptop/Documents/radar_tracker/Radar-data-processing-and-analysis/code/colreg_files"
    for file in path_list:
        new_path_list.append(os.path.join(colreg_files_dir, file))

    return new_path_list

def colreg_files_from_chosen_scenarios_txt(npy_files):
    path_list = []
    colreg_situation_files = []
    with open("chosen_scenarios.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            date = line.split(" ")[0]
            date = line.strip()     # Remove the newline character
            scenario = f"colreg_tracks_rosbag_{date}.npy"
            colreg_situation_files.append(scenario)

    temp_npy_files = []
    for file in npy_files:
        temp_npy_files.append(file.split("/")[-1])
    
    for file in temp_npy_files:
        for colreg_file in colreg_situation_files:
            if file == colreg_file:
                path_list.append(file)

    new_path_list = []
    colreg_files_dir = "/home/aflaptop/Documents/radar_tracker/Radar-data-processing-and-analysis/code/colreg_files"
    for file in path_list:
        new_path_list.append(os.path.join(colreg_files_dir, file))
    return new_path_list

def scenario_selector(import_selection):
    npy_files = list_npy_files()
    if import_selection == 0:
        return npy_files
    
    elif import_selection == 1:
        path_list = find_only_colreg_files(npy_files)
        return path_list
    
    elif import_selection == 2:
        path_list = colreg_files_from_chosen_scenarios_txt(npy_files)
        return path_list

    elif import_selection == 3:
        path_list = list_local_npy_files()
        return path_list
    else:
        print("Invalid selection")
        sys.exit(1)


if __name__ == "__main__":
    plot_statment = 0
    video_statment = 0

    # 0 all npy files saved in the colreg_files directory in the radar tracker
    # 1 only the files listed in scenarios_with_colreg_situations.txt 
    # 2 only the files listed in chosen_scenarios.txt
    import_selection = 3
    path_list = scenario_selector(import_selection)
    # print(path_list)
    # temp_in = input("Press enter to continue")

    # path_list = ["/home/aflaptop/Documents/radar_tracker/Radar-data-processing-and-analysis/code/npy_files/colreg_tracks_rosbag_2023-09-09-12-33-28.npy"]
    path_list = ["/home/aflaptop/Documents/Scripts/AIS_processing/npy_files/colreg_tracks_rosbag_2023-09-02-13-17-29_new.npy"]
    r_colregs_2_max=40    #50
    r_colregs_3_max=0     #30
    r_colregs_4_max=0     #4

    for k,data_file in enumerate(path_list):
        plot_statment = 0
        video_statment = 0
        print(f"Scenario {k+1} of {len(path_list)}")
        print(f"For file: {os.path.basename(data_file).split('.')[0].split('_')[-1]}")

        vessels = read_out_radar_data(data_file=data_file)
        AV = AutoVerification(vessels=vessels, r_colregs_2_max=r_colregs_2_max, r_colregs_3_max=r_colregs_3_max, r_colregs_4_max=r_colregs_4_max)
        AV.find_ranges()

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


        for vessel in AV.vessels:   
            AV.determine_situations(vessel)

        for vessel in AV.vessels:
            # print("Vessel id: ", vessel.id)
            # print(AV.situation_matrix[vessel.id])
            # print(np.all(AV.situation_matrix[vessel.id]))
            if all_elements_zero_or_OP(AV.situation_matrix[vessel.id]):
                print("No situations found")
            else:
                write_scenario_to_file(data_file)
                plot_statment = 1
                # video_statment = 1

        if plot_statment:
            font_size = 20
            ax, origin_x, origin_y = plotting.start_plot()
            for k,vessel in enumerate(AV.vessels):
                plotting.plot_single_vessel(vessel, ax, origin_x, origin_y)
                plotting.plot_colreg_situation(vessel, AV.situation_matrix[vessel.id], ax, origin_x, origin_y)
            save_name = f"plotting_results/plots/plot_{os.path.basename(data_file).split('.')[0].split('_')[-1]}.png"
            plt.savefig(save_name, dpi=300)
            print(f"Saved plot to {save_name}")
            # plt.show()
      
        if video_statment:
            video_object = Video(wokring_directory=os.getcwd(),filename=os.path.basename(data_file).split('.')[0].split('_')[-1])
            video_object.create_video(AV_object=AV)









