import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import sys

def start_plot():
    fig, ax = plt.subplots(figsize=(11, 7.166666))
    font_size = 20

    # Plotting the occupancy grid'
    data = np.load(f"npy_files/occupancy_grid_without_dilating.npy",allow_pickle='TRUE').item()
    occupancy_grid = data["occupancy_grid"]
    origin_x = data["origin_x"]
    origin_y = data["origin_y"]

    colors = [(1, 1, 1), (0.8, 0.8, 0.8)]  # Black to light gray
    cm = LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
    ax.imshow(occupancy_grid, cmap=cm, interpolation='none', origin='upper', extent=[0, occupancy_grid.shape[1], 0, occupancy_grid.shape[0]])
    
    ax.set_xlim(origin_x-120,origin_x + 120)
    ax.set_ylim(origin_y-140, origin_y + 20)
    ax.set_aspect('equal')
    ax.set_xlabel('East [m]',fontsize=font_size)
    ax.set_ylabel('North [m]',fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()

    # reformating the x and y axis
    x_axis_list = np.arange(origin_x-120,origin_x+121,20)
    x_axis_list_str = []
    for x in x_axis_list:
        x_axis_list_str.append(str(int(x-origin_x)))
    ax.set_xticks(ticks=x_axis_list)
    ax.set_xticklabels(x_axis_list_str)
    # plt.xticks(x_axis_list, x_axis_list_str)

    y_axis_list = np.arange(origin_y-140,origin_y+21,20)
    y_axis_list_str = []
    for y in y_axis_list:
        y_axis_list_str.append(str(int(y-origin_y)))
    ax.set_yticks(ticks=y_axis_list)
    ax.set_yticklabels(y_axis_list_str)
    # plt.yticks(y_axis_list, y_axis_list_str)

    ax.grid(True)

    return ax, origin_x, origin_y


def plot_single_vessel(vessel):
    ax, origin_x, origin_y = start_plot()
    
    x, y, psi, _, _ = vessel.state
    ax.plot(x + origin_x, y + origin_y, markersize=10)

    manuver_idx = vessel.maneuver_detect_idx
    for idx in manuver_idx:
        ax.scatter(x[idx] + origin_x, y[idx] + origin_y, color="red")
    
    manuver_start_stop = vessel.maneuver_start_stop
    # for start, stop in manuver_start_stop:
    #     ax.plot(x[start:stop] + origin_x, y[start:stop] + origin_y, color="red")

    delta_course = vessel.delta_course

    delta_speed = vessel.delta_speed
    print(len(delta_speed))
    print(len(delta_course))

    plt.show()
    # sys.exit()

