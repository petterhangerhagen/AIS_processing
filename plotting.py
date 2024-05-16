import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import sys

 # # # OP = -3  # Obstacle passed
# # # OTGW = -2  # Overtaking situation - own ship is give way vessel
# # # CRGW = -1  # Crossing situation - own ship is give way vessel
# # # NAR = 0  # No applicable rules
# # # CRSO = 1  # Crossing situation - own ship is stand on vessel
# # # OTSO = 2  # Overtaking situation - own ship is stand on vessel
# # # HO = 3  # Head on situation

colors = ['#ff7f0e','#1f77b4', '#2ca02c','#c73838','#c738c0',"#33A8FF",'#33FFBD']  # Orange, blå, grønn, rød, rosa, lyse blå, turkis

# blue_colors = ['#419ede','#1f77b4', '#144c73']
blue_colors = ['#2b93db','#1f77b4', '#1b699e']

# green_colors = ['#4bce4b','#2ca02c', '#1c641c']
green_colors = ['#32b432','#2ca02c', '#278c27']


situation_dict = {
    -3: ["OP", "Obstacle passed", colors[0]],
    -2: ["OTGW", "Overtaking give way", green_colors[0]],
    -1: ["CRGW", "Crossing give way", blue_colors[0]],
    0: ["NAR", "No applicable rules", colors[4]],
    1: ["CRSO", "Crossing stand on", blue_colors[2]],
    2: ["OTSO", "Overtaking stand on", green_colors[2]],
    3: ["HO", "Head on", colors[3]]
}

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

def plot_single_vessel_old(vessel,ax,origin_x,origin_y, color):
    
    x, y, psi, _, _ = vessel.state
    ax.plot(x + origin_x, y + origin_y, markersize=10)

    manuver_idx = vessel.maneuver_detect_idx
    for idx in manuver_idx:
        ax.scatter(x[idx] + origin_x, y[idx] + origin_y, color="red")
    
    # manuver_start_stop = vessel.maneuver_start_stop
    # # for start, stop in manuver_start_stop:
    # #     ax.plot(x[start:stop] + origin_x, y[start:stop] + origin_y, color="red")

    # delta_course = vessel.delta_course

    # delta_speed = vessel.delta_speed
    # print(len(delta_speed))
    # print(len(delta_course))

    # plt.show()
    # sys.exit()

def plot_single_vessel(vessel,ax,origin_x,origin_y, color):
    x, y, psi, _, _ = vessel.state

    # Plot the grayscale line
    timestamps = vessel.time_stamps
    # Normalize timestamps between 0 and 1
    norm = Normalize(vmin=min(timestamps), vmax=max(timestamps))
    # Convert RGBA values to grayscale
    grayscale_values = norm(timestamps)
    grayscale_values = (1 - grayscale_values) * 0.9

    # Plot the grayscale line
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2] + origin_x, y[i:i+2] + origin_y, color=(grayscale_values[i], grayscale_values[i], grayscale_values[i]), linewidth=2)

    # Plot the first point of track and annotate it
    index = vessel.nan_idx[0]
    ax.scatter(x[index] + origin_x, y[index] + origin_y, color='black',zorder=2)
    ax.annotate(f"Vessel {vessel.id}", (x[index] + origin_x + 1, y[index] + origin_y + 1), fontsize=10, color='black')
    
    legend_elements = []
    l1 = ax.scatter([], [], marker='o', c=situation_dict[-3][2], s=100, label=situation_dict[-3][0] + " - " + situation_dict[-3][1])
    l2 = ax.scatter([], [], marker='o', c=situation_dict[-2][2], s=100, label=situation_dict[-2][0] + " - " + situation_dict[-2][1])
    l3 = ax.scatter([], [], marker='o', c=situation_dict[-1][2], s=100, label=situation_dict[-1][0] + " - " + situation_dict[-1][1])
    # l4 = ax.scatter([], [], marker='o', c=situation_dict[0][2], s=100, label=situation_dict[0][0] + " - " + situation_dict[0][1])
    l5 = ax.scatter([], [], marker='o', c=situation_dict[1][2], s=100, label=situation_dict[1][0] + " - " + situation_dict[1][1])
    l6 = ax.scatter([], [], marker='o', c=situation_dict[2][2], s=100, label=situation_dict[2][0] + " - " + situation_dict[2][1])
    l7 = ax.scatter([], [], marker='o', c=situation_dict[3][2], s=100, label=situation_dict[3][0] + " - " + situation_dict[3][1])

    legend_elements.append(l1)
    legend_elements.append(l2)
    legend_elements.append(l3)
    # legend_elements.append(l4)
    legend_elements.append(l5)
    legend_elements.append(l6)
    legend_elements.append(l7)
        
    font_size = 17
    ax.legend(handles=legend_elements, fontsize=font_size, loc='lower right')


def plot_colreg_situation(vessel, situation_matrix, ax, origin_x, origin_y, color):
    x, y, psi, _, _ = vessel.state

    situation_matrix = situation_matrix.copy()
    for i in range(len(situation_matrix)):
        # Skip the vessel itself
        if i == vessel.id:
            continue
        # Skip if there is no situation
        if np.all(situation_matrix[i] == 0):
            print(f"No situation with vessel {i}")
            continue

        # Get the indices of the situation and plot them
        indices = np.nonzero(situation_matrix[i])[0]
        for idx in indices:
            color = situation_dict[situation_matrix[i][idx]][2]
            ax.scatter(x[idx] + origin_x, y[idx] + origin_y, color=color, zorder=2)
    



