"""
Script Title: radar_data_main.py
Author: Petter Hangerhagen
Email: petthang@stud.ntnu.no
Date: July 4th, 2024
Description: This is the main file in this codebase. It reads the radar data and creates a list of vessel objects, which is the input to the AutoVerification class.
From the radar-based dataset all the scenarios containing more than one vessel track is included in the colreg_complete_dataset directory.
This scripts reads different scenarios from the colreg_complete_dataset directory and creates a list of vessel objects form the radar_data_to_vessels.py file.
"""

import numpy as np
import plotting
import sys
import matplotlib.pyplot as plt
import os
from video import Video
import utilities as utils
import radar_data_to_vessels

def scenario_selector(import_selection, colreg_files_dir):
    npy_files = utils.list_npy_files(colreg_files_dir)

    # All npy files in the colreg_files_dir
    if import_selection == 0:
        return npy_files
    
    # Only the files listed in scenarios_with_colreg_situations.txt
    elif import_selection == 1:
        path_list = utils.find_only_colreg_files(npy_files, colreg_files_dir)
        return path_list
    
    # Only the files listed in chosen_scenarios.txt
    elif import_selection == 2:
        path_list = utils.colreg_files_from_chosen_scenarios_txt(npy_files, colreg_files_dir)
        return path_list

    # All npy files saved in the npy_files directory
    elif import_selection == 3:
        path_list = utils.list_local_npy_files()
        return path_list
    else:
        print("Invalid selection")
        sys.exit(1)

situation_dict = {
    -3: 0,      # Obstacle passed
    0: 0,       # No applicable rules
    -2: 0,      # Overtaking give way
    2: 0,       # Overtaking stand on
    3: 0,       # Head on
    -1: 0,      # Crossing give way
    1: 0        # Crossing stand on
}

if __name__ == "__main__":
    # Get the file path to the dataset
    working_dir = os.getcwd()
    colreg_files_dir = "colreg_complete_dataset"
    colreg_files_dir = os.path.join(working_dir, colreg_files_dir)

    # Turn on/off different functionalities
    plot_statment = 1
    video_statment = 0
    count_number_of_situations = 0
    add_CPA = 1

    """
    Import selection:
    0 - all npy files saved save in dataset directory
    1 - only the files listed in scenarios_with_colreg_situations.txt 
    2 - only the files listed in chosen_scenarios.txt
    3 - all npy files saved in the npy_files directory
    """
    import_selection = 3
    path_list = scenario_selector(import_selection, colreg_files_dir)
  
    r_colregs_2_max=100   #50
    r_colregs_3_max=0     #30
    r_colregs_4_max=0     #4

    # # SHIP DOMAIN PLOT
    # plotting.plot_ship_domain(radius=r_colregs_2_max)
    # plt.savefig("plotting_results/ship_domain.png", dpi=300)
    # plt.show()
    # sys.exit(1)

    # HISOGRAM PLOT OF SITUATIONS
    # situation_dict = np.load("situation_dict.npy", allow_pickle=True).item()
    # plotting.plot_histogram_situations(situation_dict)
    # plt.savefig("plotting_results/histogram_situations.png", dpi=300)
    # # plt.show()
    # sys.exit(1)

    total_num_situations = 0
    num_of_scenarios_without_situations = 0
    for k,data_file in enumerate(path_list):
        print(f"Scenario {k+1} of {len(path_list)}")
        print(f"For file: {os.path.basename(data_file).split('.')[0].split('_')[-1]}")
      
        vessels = radar_data_to_vessels.read_out_radar_data(data_file=data_file)
        AV = AutoVerification(vessels=vessels, r_colregs_2_max=r_colregs_2_max, r_colregs_3_max=r_colregs_3_max, r_colregs_4_max=r_colregs_4_max)
        AV.find_ranges()

        for vessel in AV.vessels:
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

       
        if count_number_of_situations:
            number_of_sit = 0
            for vessel in AV.vessels:
                number_of_sit += utils.count_scenarios(AV.situation_matrix[vessel.id], situation_dict)
            if number_of_sit == 0:
                num_of_scenarios_without_situations += 1
            total_num_situations += number_of_sit
            print(f"Number of situations: {number_of_sit}")

        if add_CPA:
            min_distance, index_of_cpa = utils.closest_point_of_approach(AV.vessels)

        if plot_statment:
            font_size = 20
            ax, origin_x, origin_y = plotting.start_plot()
            for k,vessel in enumerate(AV.vessels):
                plotting.plot_single_vessel(vessel, ax, origin_x, origin_y)
                plotting.plot_colreg_situation(vessel, AV.situation_matrix[vessel.id], ax, origin_x, origin_y)
            
            if add_CPA:
                if min_distance != None:
                    ax.plot(np.array([AV.vessels[0].state[0, index_of_cpa],AV.vessels[1].state[0, index_of_cpa]]) + origin_x, np.array([AV.vessels[0].state[1, index_of_cpa],AV.vessels[1].state[1, index_of_cpa]])+ origin_y, color='black', linestyle='--')
                    ax.annotate(f"CPA: {min_distance:.2f} m", (AV.vessels[0].state[0, index_of_cpa] + origin_x + 1, 0 + origin_y + 1), fontsize=font_size, color='black')
            save_name = f"plotting_results/plots/plot_{os.path.basename(data_file).split('.')[0].split('_')[-1]}.png"
            plt.savefig(save_name, dpi=300)
            print(f"Saved plot to {save_name}")
            plt.show()
            plt.close()
      
        if video_statment:
            video_object = Video(wokring_directory=os.getcwd(),filename=os.path.basename(data_file).split('.')[0].split('_')[-1])
            video_object.create_video(AV_object=AV)
        

    print("Done")
    print(f"Total number of situations: {total_num_situations}")
    print(f"Number of scenarios without situations: {num_of_scenarios_without_situations}")
    if count_number_of_situations:
        np.save("situation_dict.npy", situation_dict)







