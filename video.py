"""
Script Title: Video
Author: Petter Hangerhagen and Audun Gullikstad Hem
Email: petthang@stud.ntnu.no
Date: February 27, 2024
Description: This script is not directly from Audun Gullikstad Hem mulit-target tracker (https://doi.org/10.24433/CO.3351829.v1), but it is inspired by the plotting script which is from the tracker. 
It is used to create a video of the tracking scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import progressbar
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm  # Import the colormap module
from images_to_video import images_to_video_opencv, empty_folder

# define font size, and size of plots
matplotlib.rcParams['font.size'] = 20

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

class Video(object):
    """
    A class representing a plot depicitng the tracking scenario.
    """
    def __init__(self, wokring_directory, filename="coord_69",resolution=100,fps=1):
        self.wokring_directory = wokring_directory
        self.filename = filename.split("/")[-1].split("_")[-1].split(".")[0]
        self.resolution = 100
        self.fps = fps
        self.start_plot()
        self.plot_count = 2
        self.font_size = 17
        self.read_out_file()

    def start_plot(self):
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

        self.fig = fig
        self.ax = ax
        self.origin_x = origin_x
        self.origin_y = origin_y

    def create_legend(self):
        legend_elements = []
        l1 = self.ax.scatter([], [], marker='o', c=situation_dict[-3][2], s=100, label=situation_dict[-3][0] + " - " + situation_dict[-3][1])
        l2 = self.ax.scatter([], [], marker='o', c=situation_dict[-2][2], s=100, label=situation_dict[-2][0] + " - " + situation_dict[-2][1])
        l3 = self.ax.scatter([], [], marker='o', c=situation_dict[-1][2], s=100, label=situation_dict[-1][0] + " - " + situation_dict[-1][1])
        l5 = self.ax.scatter([], [], marker='o', c=situation_dict[1][2], s=100, label=situation_dict[1][0] + " - " + situation_dict[1][1])
        l6 = self.ax.scatter([], [], marker='o', c=situation_dict[2][2], s=100, label=situation_dict[2][0] + " - " + situation_dict[2][1])
        l7 = self.ax.scatter([], [], marker='o', c=situation_dict[3][2], s=100, label=situation_dict[3][0] + " - " + situation_dict[3][1])
        legend_elements.append(l1)
        legend_elements.append(l2)
        legend_elements.append(l3)
        legend_elements.append(l5)
        legend_elements.append(l6)
        legend_elements.append(l7)
        self.legend_elements = legend_elements

    def create_video(self, AV_object):
        
        
        vessels = AV_object.vessels
        timestamps = vessels[0].time_stamps

        # Progress bar
        bar = progressbar.ProgressBar(maxval=len(timestamps)).start()

        # Normalize timestamps between 0 and 1
        norm = Normalize(vmin=min(timestamps), vmax=max(timestamps))
        # Convert RGBA values to grayscale
        grayscale_values = norm(timestamps)
        grayscale_values = (1 - grayscale_values) * 0.8

        self.create_legend()
        self.ax.legend(handles=self.legend_elements, fontsize=12, loc='lower right')
        
        start_dict = {} 
        for vessel in vessels:
            start_dict[vessel.id] = list(vessel.nan_idx[0])
    
        line_colors = []
        situation_matrix = AV_object.situation_matrix
        while self.plot_count < len(timestamps):
            # print(f"Plotting frame {self.plot_count} of {len(timestamps)}")
            for k,vessel in enumerate(vessels):
                x, y, _, _, _ = vessel.state
                x = np.array(x) + self.origin_x
                y = np.array(y) + self.origin_y

                if self.plot_count-2 == start_dict[vessel.id][0]:
                    # print(f"Adding start point for vessel {vessel.id}")
                    index = start_dict[vessel.id][0]
                    self.ax.scatter(x[index], y[index], color='black', zorder=10)
                    self.ax.annotate(f"Vessel {vessel.id}", (x[index] + 1, y[index] + 1), fontsize=10, color='black')

                line_color = cm.gray(grayscale_values[self.plot_count])
                line_colors.append(line_color)
                self.ax.plot(x[self.plot_count-2:self.plot_count], y[self.plot_count-2:self.plot_count], color=line_colors[self.plot_count-2], linewidth=2)

                current_situation_matrix = situation_matrix[vessel.id]
                for i in range(len(current_situation_matrix)):
                    if i == vessel.id:
                        continue
                    current_situation = current_situation_matrix[i][self.plot_count-2]
                    if current_situation == 0:
                        continue
                    # print(f"Vessel {vessel.id} has a situation with vessel {i}")
                    color = situation_dict[current_situation][2]
                    self.ax.scatter(x[self.plot_count-2], y[self.plot_count-2], color=color, zorder=2)

            # Saving the frame
            self.ax.set_title(f"Time: {timestamps[self.plot_count]:.2f} s", fontsize=10)
            # self.write_to_video()

            temp_save_path = f'{self.wokring_directory}/plotting_results/videos/temp/tracker_{self.plot_count}.png'
            self.fig.savefig(temp_save_path,dpi=self.resolution)
            self.plot_count += 1

            bar.update(self.plot_count)

        # Saving the video
        photos_file_path = f'{self.wokring_directory}/plotting_results/videos/temp'
        video_name = f'{photos_file_path[:-4]}{self.filename}.avi'
        images_to_video_opencv(photos_file_path, video_name, self.fps)
        print(f"\nSaving the video to {video_name}")
        empty_folder(photos_file_path)
       
    def write_to_video(self):
        # beta, beta_180, alpha, alpha_360 = AV_object.angles
        for txt in self.ax.texts:
            txt.remove() 
        text = ""
        for row in self.angle_data:
            
            # self.ax.text(0.0, -0.13, "", transform=self.ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.15))
            if self.plot_count-2 == row[2]:
                vessel_id = row[0]
                object_id = row[1]
                # print(f"Vessel {vessel_id}, object {object_id}")
                beta, beta_180, alpha, alpha_360 = row[3:7]
                text += f"Vessel {vessel_id}, object {object_id}: "
                # text += f"$\beta$: {beta:.2f}, beta_180: {beta_180:.2f}, alpha: {alpha:.2f}, alpha_360: {alpha_360:.2f}\n"
                text += fr"$\beta$: {beta:.2f}, $\beta_{{180}}$: {beta_180:.2f}, $\alpha$: {alpha:.2f}, $\alpha_{{360}}$: {alpha_360:.2f}"
                text += "\n"

        if text != "":
            # print(text)
            self.ax.text(0.0, -0.2, text, transform=self.ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.15))
            # plt.pause(1)
        else:
            self.ax.text(0.0, -0.2, "\n \n", transform=self.ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.15))
        plt.tight_layout()

    def read_out_file(self):
        # Initialize an empty list to store the data
        data_matrix = []

        # Open the file
        with open('angles.txt', 'r') as file:
            # Read each line in the file
            for line in file:
                # Split the line by commas to get individual fields
                fields = line.strip().split(',')
                # Convert fields to floats and append them to the data_matrix
                data_matrix.append([float(field.strip()) for field in fields])

        self.angle_data = data_matrix

