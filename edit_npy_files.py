import numpy as np
import matplotlib.pyplot as plt
from plotting import start_plot

def constant_velocity_model(prev_value, current_value, position_std_dev):
    prev_timestamp, prev_x, prev_y, prev_psi, prev_x_vel, prev_y_vel = prev_value
    timestamp, x, y, psi, x_vel, y_vel = current_value
    
    dt = timestamp - prev_timestamp  # Time difference
    
    # Calculate new positions based on constant velocity model
    new_x = prev_x + prev_x_vel*0.15 * dt
    new_y = prev_y + prev_y_vel*0.15 * dt
    
    # Add Gaussian noise to introduce uncertainty
    new_x += np.random.normal(0, position_std_dev)
    new_y += np.random.normal(0, position_std_dev)
    
    # Return the new value with updated positions
    return (timestamp, new_x, new_y, psi, prev_x_vel, prev_y_vel)


npy_file_name = "npy_files/colreg_tracks_rosbag_2023-09-02-13-17-29.npy"
data = np.load(npy_file_name, allow_pickle=True).item()

ax, origin_x, origin_y = start_plot()

x_limit = -60
y_limit = -115

position_std_dev = 0.05

new_data = {}
for key, values in data.items():
    new_data[key] = []
    xs = []
    ys = []
    prev_value = None
    if key == 0:
        for value in values:
            if prev_value is None:
                new_value = value
            else:
                timestamp, _, _, _, _, _ = value
                if 22 < timestamp < 131:
                    new_value = constant_velocity_model(prev_value, value, position_std_dev)
                else:
                    # Use the original value
                    new_value = value
            # Unpack the new value
            timestamp, x, y, psi, x_vel, y_vel = new_value
            new_data[key].append(new_value)
            xs.append(x + origin_x)
            ys.append(y + origin_y)
            prev_value = new_value
    else:
        for value in values:
            _, x, y, _, _, _ = value
            new_data[key].append(value)
            xs.append(x + origin_x)
            ys.append(y + origin_y)
    ax.plot(xs, ys, label=key)

    # if key == 0:
    #     for value in values:
    #         new_value = value
    #         timestamp, _, _, _, _, _ = value
    #         if 36 < timestamp < 131:
    #             new_value = constant_velocity_model(prev_value,new_value)
    #             # continue
    #         timestamp, x, y, psi, x_vel, y_vel = new_value
    #         new_data[key].append(new_value)
    #         xs.append(x + origin_x)
    #         ys.append(y + origin_y)
    #         prev_value = new_value
        
    # else:
    #     for value in values:
    #         _, x, y, _, _, _ = value
    #         new_data[key].append(value)
    #         xs.append(x + origin_x)
    #         ys.append(y + origin_y)
    #     ax.plot(xs, ys, label=key)



# name = npy_file_name.split(".")[0]
# np.save(f"{name}_new", new_data)

ax.legend()
plt.show()
        