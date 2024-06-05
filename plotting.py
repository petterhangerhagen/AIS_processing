"""
Script Title: plotting.py
Author: Petter Hangerhagen
Email: petthang@stud.ntnu.no
Date: June 4th, 2024
Description: This script contains functions used to plot the colreg situations and the vessel tracks.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

colors = ['#ff7f0e','#1f77b4', '#2ca02c','#c73838','#c738c0',"#33A8FF",'#33FFBD']  # Orange, blå, grønn, rød, rosa, lyse blå, turkis
blue_colors = ['#2b93db','#1f77b4', '#1b699e']
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

# Function that defines the base plot
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

# Function to plot a single vessel track, also adds the legend with the different situations from the situation_dict
def plot_single_vessel(vessel,ax,origin_x,origin_y):
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
        ax.plot(x[i:i+2] + origin_x, y[i:i+2] + origin_y, color=(grayscale_values[i], grayscale_values[i], grayscale_values[i]), linewidth=2, zorder=3)

    # Plot the first point of track and annotate it
    index = vessel.nan_idx[0]
    ax.scatter(x[index] + origin_x, y[index] + origin_y, color='black',zorder=10)
    ax.annotate(f"Start vessel {vessel.id}", (x[index] + origin_x + 1, y[index] + origin_y + 1), fontsize=10, color='black')
    
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

# Function to plot the colreg situations
def plot_colreg_situation(vessel, situation_matrix, ax, origin_x, origin_y):
    x, y, psi, _, _ = vessel.state

    situation_matrix = situation_matrix.copy()
    for i in range(len(situation_matrix)):
        already_passed = False
        # Skip the vessel itself
        if i == vessel.id:
            continue
        # Skip if there is no situation
        if np.all(situation_matrix[i] == 0):
            # print(f"No situation with vessel {i}")
            continue

        # Get the indices of the situation and plot them
        indices = np.nonzero(situation_matrix[i])[0]
        for idx in indices:
            color = situation_dict[situation_matrix[i][idx]][2]
            if situation_matrix[i][idx] == -3:
                if already_passed:
                    continue
                else:
                    ax.scatter(x[idx] + origin_x, y[idx] + origin_y, color=color, alpha=1, zorder=2)
                    already_passed = True
            else:
                ax.scatter(x[idx] + origin_x, y[idx] + origin_y, color=color, alpha=1, zorder=2)
                already_passed = False

# Function used to plot the ship domain, which is used in the thesis
def plot_ship_domain(radius):
    ax, origin_x, origin_y = start_plot()
    center = (0 + origin_x, -50 + origin_y)
    ax.scatter(center[0], center[1], color="black", marker='o', zorder=10)
    ax.annotate(r"$\mathbf{r}_{{colreg}} = 100$", (center[0] -4, center[1] + 20), fontsize=17, color='black')
    angle = 60
    ax.quiver(center[0], center[1], radius*np.sin(np.deg2rad(angle)),radius*np.cos(np.deg2rad(angle)), color='black', scale=1, scale_units='xy', zorder=10)
    circle = plt.Circle(center, radius, color='black', fill=False, linewidth=2)
    ax.add_artist(circle)

# Function to plot a histogram of the situations
def plot_histogram_situations(situation_dict_in):
    fig, ax = plt.subplots(figsize=(11, 7.166666))
    font_size = 20
    named_dict = {
        1: ["CR","Crossing","#1f77b4"],
        2: ["OT","Overtaking", "#2ca02c"],
        3: ["HO","Head on", "#c73838"],
    }

    # Plotting the histogram
    for key, value in situation_dict_in.items():
        if key == -3 or key==-2 or key==-1 or key == 0 :
            continue
        bar = ax.bar(named_dict[key][0], value, width=0.6, color=named_dict[key][2], label=named_dict[key][0] + " - " + named_dict[key][1])
        
        # Adding the value on top of the bar
        for rect in bar:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0, height, str(value),
                ha='center', va='bottom', fontsize=font_size
            )
    ax.legend(fontsize=font_size)
    ax.set_xlabel('Situations', fontsize=font_size)
    ax.set_ylabel('Number of Situations', fontsize=font_size)
    ax.set_ylim(0, max(situation_dict_in.values())*1.1)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
