import numpy as np
from AutoVerification import AutoVerification, Vessel
from scipy.ndimage.filters import gaussian_filter

# The message need to be of this form [x y yaw u v]
vessels = []
data = np.load("colreg_tracks_old.npy", allow_pickle=True).item()

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
               
    vessel = Vessel(vessel_id, len(total_timestamps))
    vessel.id = vessel_id
    vessel.stateDateTime = total_timestamps
    vessel.state[0] = x_positions_new
    vessel.state[1] = y_positions_new
    vessel.state[2] = yaws_new
    vessel.state[3] = x_velocities_new
    vessel.state[4] = y_velocities_new

    sog = np.sqrt(np.square(vessel.state[3]) + np.square(vessel.state[4]))
    vessel.speed = sog
    dt = np.diff(total_timestamps)
    dt = np.append(dt, dt[-1])
    dt = np.mean(dt)
    vessel.dT = dt
    

    # sog = np.sqrt(np.square(x_velocities) + np.square(y_velocities))
    # vessel.speed = sog
    # print(vessel.speed)
    # print("\n")

    


    # print(vessel_id)
    # print(timestamps)
    # print("\n")
#     vessel = Vessel(vessel_id, len(timestamps))
#     vessel.id = vessel_id
#     vessel.stateDateTime = timestamps
#     vessel.state[0] = x_positions
#     vessel.state[1] = y_positions
#     vessel.state[2] = yaws
#     vessel.state[3] = x_velocities
#     vessel.state[4] = y_velocities
    

#     sog = np.sqrt(np.square(x_velocities) + np.square(y_velocities))
#     vessel.speed = sog
#     dt = np.diff(timestamps)
#     dt = np.append(dt, dt[-1])
#     dt = np.mean(dt)
#     vessel.dT = dt


#     vessel.travel_dist = np.linalg.norm(np.array([x_positions[-1], y_positions[-1]]) - np.array([x_positions[0], y_positions[0]]))

#     if vessel.travel_dist > 1000:
#         # Calculate derivative of speed
#         speed = np.array(vessel.speed)
#         speed_smoothed = gaussian_filter(speed, sigma=1)
        
#         speed_derivative = np.gradient(speed_smoothed)
        
#         vessel.speed_der[:] = speed_derivative

#         # Calculate derivatives of yaw
#         a = np.array(vessel.state[2, :])
#         d = np.append([0], np.diff(a), 0)
#         d[abs(d) < np.pi] = 0
#         d[d < -np.pi] = -2 * np.pi
#         d[d > np.pi] = 2 * np.pi
        
#         s = np.cumsum(d, axis=0)
        
#         a_smoothed = gaussian_filter(a, sigma=2)
        
#         a_derivative = np.gradient(a_smoothed)
        
#         vessel.maneuver_der[0, :] = np.gradient(a_derivative)
#         vessel.maneuver_der[1, :] = np.gradient(np.gradient(a_derivative))
#         vessel.maneuver_der[2, :] = np.gradient(np.gradient(np.gradient(a_derivative)))

#     vessels.append(vessel)


# for vessel in vessels:
#     print(f"vessel id = {vessel.id}")
#     # print(vessel.state)
#     # print(vessel.speed)
#     # print(vessel.travel_dist)
#     # print("\n")


# # r_colregs_2_max=5000
# # r_colregs_3_max=3000
# # r_colregs_4_max=400
# r_colregs_2_max=50
# r_colregs_3_max=30
# r_colregs_4_max=4

# AV = AutoVerification(vessels=vessels, r_colregs_2_max=r_colregs_2_max, r_colregs_3_max=r_colregs_3_max, r_colregs_4_max=r_colregs_4_max)

# # Can not run this since the vessel states have different format
# # Wondering if i need to add in nan values for the missing data
# AV.find_ranges()