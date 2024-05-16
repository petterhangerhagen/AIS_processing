import numpy as np
from AutoVerification import AutoVerification, Vessel
from scipy.ndimage.filters import gaussian_filter
import plotting
import sys
import matplotlib.pyplot as plt

def read_out_radar_data():
    # The message need to be of this form [x y yaw u v]
    vessels = []
    data = np.load("npy_files/colreg_tracks_old.npy", allow_pickle=True).item()

    timestamps_dict = {}

    for vessel_id, track in data.items():
        if vessel_id == 1:
            continue
        timestamps, x_positions, y_positions, yaws, x_velocities, y_velocities = zip(*track)
        timestamps_dict[vessel_id] = timestamps

    total_timestamps = set()
    for vessel_id, timestamps in timestamps_dict.items():
        total_timestamps.update(timestamps)
    total_timestamps = sorted(list(total_timestamps))
    print(len(total_timestamps))

    for vessel_id, track in data.items():
        if vessel_id == 1:
            continue
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
            vessel.state[2, j] = yaws_new[j]
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
            
           

        # vessel.state[0] = x_positions_new
        # vessel.state[1] = y_positions_new
        # vessel.state[2] = yaws_new
        # vessel.state[3] = x_velocities_new
        # vessel.state[4] = y_velocities_new

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

                from scipy.ndimage.filters import gaussian_filter
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
                    

        vessels.append(vessel)
        for id_idx, vessel in enumerate(vessels):
            vessel.id = id_idx
    return vessels

def find_ranges(vessels):
    n_msgs = vessels[0].n_msgs
    n_vessels = len(vessels)
    ranges = np.zeros([n_vessels, n_vessels, n_msgs], dtype=float)
    # print(ranges)
    # print(ranges.shape)
    ranges_set = False
    detection_idx = np.zeros([n_vessels, n_vessels], dtype=int)
    r_detect = 50  
    cpa_idx = np.zeros([n_vessels, n_vessels], dtype=int)

    for vessel in vessels:
        for obst in vessels:
            if vessel.id < obst.id:
                for i in range(n_msgs):
                    print(vessel.id, obst.id, i)
                    ranges[(vessel.id, obst.id), (obst.id, vessel.id), i] = \
                        np.linalg.norm(vessel.state[0:2, i] - obst.state[0:2, i])
                detection_idx[(vessel.id, obst.id), (obst.id, vessel.id)] = \
                    np.argmax(ranges[vessel.id, obst.id] <= r_detect)
                r_cpa = np.min(ranges[vessel.id, obst.id])
                cpa_idx[(vessel.id, obst.id), (obst.id, vessel.id)] = \
                    np.argmax(ranges[vessel.id, obst.id] == r_cpa) 
    ranges_set = True

if __name__ == "__main__":

    r_colregs_2_max=50
    r_colregs_3_max=30
    r_colregs_4_max=4

    vessels = read_out_radar_data()
    AV = AutoVerification(vessels=vessels, r_colregs_2_max=r_colregs_2_max, r_colregs_3_max=r_colregs_3_max, r_colregs_4_max=r_colregs_4_max)
    AV.find_ranges()

    for vessel in AV.vessels:
        AV.find_maneuver_detect_index(vessel)  # Find maneuvers made by ownship
        print("##############################################")
        print(f"Vessel id = {vessel.id}")
        print("##############################################")
        print("Maneuver detect index:")
        print(vessel.maneuver_detect_idx)
        print("\n")
        print("Maneuver start stop:")
        print(vessel.maneuver_start_stop )
        print("\n")
        print("Delta course:")
        print(vessel.delta_course)
        print("\n")
        print("Delta speed:")
        print(vessel.delta_speed )
        # plotting.plot_single_vessel(vessel)



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


    colors = ['#ff7f0e','#1f77b4', '#2ca02c','#c73838','#c738c0',"#33A8FF",'#33FFBD']  # Orange, blå, grønn, rød, rosa, lyse blå, turkis
    font_size = 20
    ax, origin_x, origin_y = plotting.start_plot()
    for k,vessel in enumerate(AV.vessels):
        plotting.plot_single_vessel(vessel, ax, origin_x, origin_y, colors[k])
        plotting.plot_colreg_situation(vessel, AV.situation_matrix[vessel.id], ax, origin_x, origin_y, colors[k])
    
    plt.show()









