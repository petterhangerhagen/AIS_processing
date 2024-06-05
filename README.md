# COLREG Situations identification Algorithm

## Overview
This code is used in my master thesis. It is a simplified version of the code developed by Inger Hagen. It uses radar data as input and finds COLREG situations in the data. It identifies head-on situations, overtaking situations and crossing situations. 

## Dataset
The dataset used is recorded from an onshore sensor station in Trondheim the summer of 2023. The dataset can be obtained from https://zenodo.org/doi/10.5281/zenodo.10706215. A multi-target tracker is used on the dataset to establish tracks, these are then given as input to this code. Only scenarios containing more then one valid track are considered. 

## Code files
The main script is `radar_data_main.py`, which depends on 
- `utilities.py`
- `check_start_and_stop.py`
- `GMM_components.py`.

`plotting.py` is used to create plots. This scripts is a mess. Recommend to create own code for this. 


## Installation
To run the code, you need Python 3.8 or later with the following packages installed:
- NumPy
- Matplotlib
- Scikit-learn

There are maybe some more packages which are needed, but just install the ones needed depending on the error codes. 

