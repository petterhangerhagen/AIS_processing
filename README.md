# COLREG Situations identification Algorithm

## Overview
This code is used in my master thesis. It is a simplified version of the code developed by Inger Hagen. It uses radar data as input and finds COLREG situations in the data. It identifies head-on situations, overtaking situations and crossing situations. 

## Dataset
The dataset used is recorded from an onshore sensor station in Trondheim the summer of 2023. The dataset can be obtained from https://zenodo.org/doi/10.5281/zenodo.10706215. A multi-target tracker is used on the dataset to establish tracks (https://doi.org/10.24433/CO.3351829.v1), these are then given as input to this code. Only scenarios containing more then one valid track are considered. 

## Code files
The main script is `radar_data_main.py`, which depends on 
- `utilities.py`
- `radar_data_to_vessels.py`
- `AutoVerification.py`.
- `AV_class_functions/helper_methods.py`, support to `AutoVerification.py`.

These four scripts are the ones which are able to identify COLREG situations.

Other scripts:
- `plotting.py` is used to create plots for my thesis.
- `video.py` and `images_to_video.py` is used to create videos, which was great for debugging.
- `edit_npy_files.py` was used to remove unnecessary tracks for cleaner plots.

## Directories
- `colreg_complete_dataset` contains 345 scenarios where COLREG situations are tried indentified.
- `npy_files` contains modifed scenarios, the occupancy grid, and a situations dict (stores info about the scenarios)
- `plotting_results`, directory where plots and videos are saved.
- `txt_files`, directory where txt files are saved.


## Installation
To run the code, you need Python 3.8 or later with the following packages installed:
- NumPy
- Matplotlib
- Scikit-learn

There are maybe some more packages which are needed, but just install the ones needed depending on the error codes. 

