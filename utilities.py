"""
Script Title: utilities.py
Author: Petter Hangerhagen
Email: petthang@stud.ntnu.no
Date: July 4th, 2024
Description: This file contains utility functions used in the radar_data_main.py file. 
Many of the functions are used to read and save scenarios from the colreg dataset.
One function is used to count the number of colreg situations in a scenario.
Another function is used to find the closest point of approach between two vessels.
"""

import numpy as np
import os

# Function use to write the scenario name of scenarios with colreg situations to a txt file
def write_scenario_to_file(data_file):
    data_file = data_file.split("/")[-1]

    txt_file = "txt_files/scenarios_with_colreg_situations.txt"
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

# Function to read all the scenarios for the colreg dataset
def list_npy_files(colreg_files_dir):
    directory = colreg_files_dir
    npy_files = [file for file in os.listdir(directory) if file.endswith('.npy')]
    new_files = []
    for file in npy_files:
        new_files.append(os.path.join(directory, file))
    return new_files

# Function to read out the scenarios with colreg situations defined in the scenarios_with_colreg_situations.txt file
def find_only_colreg_files(npy_files, colreg_files_dir):
    path_list = []
    colreg_situation_files = []
    with open("txt_files/scenarios_with_colreg_situations.txt", "r") as f:
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
    for file in path_list:
        new_path_list.append(os.path.join(colreg_files_dir, file))

    return new_path_list

# Function to read out the chosen scenarios from situations_chosen_for_illustration.txt file
def colreg_files_from_chosen_scenarios_txt(npy_files, colreg_files_dir):
    path_list = []
    colreg_situation_files = []
    with open("txt_files/situations_chosen_for_illustration.txt", "r") as f:
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
    for file in path_list:
        new_path_list.append(os.path.join(colreg_files_dir, file))
    return new_path_list

# Function to read out all the scenarios saved in the npy_files directory. These are modifed scenarios, where unessecary tracks has been removed
def list_local_npy_files():
    current_dir = os.getcwd()
    colreg_files_dir = os.path.join(current_dir, "npy_files/situations_chosen_for_illustration")
    # colreg_files_dir = "/home/aflaptop/Documents/Scripts/AIS_processing/npy_files"
    directory = colreg_files_dir
    npy_files = [file for file in os.listdir(directory) if file.startswith('colreg_tracks_rosbag') and file.endswith('.npy')]
    new_files = []
    for file in npy_files:
        new_files.append(os.path.join(directory, file))
    return new_files

# Function to count scenarios
def count_scenarios(matrix, situation_dict):
    number_of_sit = 0
    for row in matrix:
        current_state = 0  # Start with no applicable rules
        for value in row:
            if value != current_state and value != 0 and value != -3:
                situation_dict[value] += 1
                number_of_sit += 1
                current_state = value
    return number_of_sit

# Function to find the closest point of approach of two vessels
def closest_point_of_approach(vessels):
    prev_distance = 1000
    index_num = None
    for vessel in vessels:
        for obst in vessels:
            if vessel.id == obst.id:
                continue
            for k in range(vessel.n_msgs):
                if np.isnan(vessel.state[0, k]) or np.isnan(obst.state[0, k]):
                    continue
                distance = np.linalg.norm(
                    [vessel.state[0, k] - obst.state[0, k], vessel.state[1, k] - obst.state[1, k]])
                if distance < prev_distance:
                    prev_distance = distance
                    index_num = k
    if index_num == None:
        print("No CPA found")
        return None, None
    else:
        return prev_distance, index_num + 0
    
