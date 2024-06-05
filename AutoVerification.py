"""
Script Title: radar_data_main.py
Author: Inger Hagen and Petter Hangerhagen
Email: petthang@stud.ntnu.no
Date: July 4th, 2024
Description: This code is developed by Inger Hagen, it is modifed a bit by Petter Hangerhagen to be able to take radar data as input and not AIS.
All functions not needed in the original code are removed.
"""

import numpy as np
import pandas as pd
import datetime
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import warnings
from AV_class_functions.helper_methods import *

from matplotlib import cm
from dataclasses import dataclass
from matplotlib.colors import LinearSegmentedColormap


warnings.filterwarnings("ignore")

"""
The difference between this code and Inger Hagen's code is:
    - __init__
    - determine_applicable_rules
"""

class AutoVerification:
    # "Constants"
    OWN_SHIP = 0  # index of own_ship

    OP = -3  # Obstacle passed
    OTGW = -2  # Overtaking situation - own ship is give way vessel
    CRGW = -1  # Crossing situation - own ship is give way vessel
    NAR = 0  # No applicable rules
    CRSO = 1  # Crossing situation - own ship is stand on vessel
    OTSO = 2  # Overtaking situation - own ship is stand on vessel
    HO = 3  # Head on situation

    def __init__(self,
                # # #  ais_path=[],
                # # #  ship_path=[],
                 vessels,
                 r_colregs_2_max=5000,
                 r_colregs_3_max=3000,
                 r_colregs_4_max=400,
                 epsilon_course=4,
                 epsilon_speed=0.5,
                 alpha_critical_13=45.0,
                 alpha_critical_14=30.0,
                 alpha_critical_15=0.0,
                 phi_OT_min=112.5,
                 phi_OT_max=247.5,
                 phi_SB_lim=-10.0):

        """
        :param ais_path: relative path to .csv file containing encounter data
        :type ais_path: str
        :param ship_path: relative path to .csv file containing ship information
        :type ship_path: str
        :param r_colregs_2_max: [m] Maximum range for COLREGS stage 2.
        :type r_colregs_2_max: int
        :param r_colregs_3_max: [m] Maximum range for COLREGS stage 3.
        :type r_colregs_3_max: int
        :param r_colregs_4_max: [m] Maximum range for COLREGS stage 4. Usually four ship lengths.
        :type r_colregs_4_max: int

        :param epsilon_course: Detectable course change. [deg/s]
        :type epsilon_course: float
        :param epsilon_speed: Detectable speed change. [m/s^2]
        :type epsilon_speed: float

        :param delta_chi_md: Minimum detectable course change [deg]
        :type delta_chi_md: float
        :param delta_psi_md: Minimum detectable heading change [deg]
        :type delta_psi_md: float
        :param delta_speed_md: Minimum detectable speed change [m/s^2]
        :type delta_speed_md: float
        
        :param alpha_critical_13: Angle defining an overtaking situation, when a vessel is approaching another from
        abaft the beam cf. rule 13.
        :type alpha_critical_13: float

        :param alpha_critical_14: Angle defining a head-on situation, when two vessels are approaching on reciprocal or
        nearly reciprocal courses cf. rule 14.
        :type alpha_critical_14: float

        :param alpha_critical_15: Angle defining a crossing situation cf. rule 15.
        :type alpha_critical_15: float
        """


        self.vessels = vessels
        self.n_vessels = len(vessels)
        self.n_msgs = self.vessels[0].n_msgs

        self.r_colregs = [r_colregs_2_max, r_colregs_3_max, r_colregs_4_max]
        self.r_detect = r_colregs_2_max  # Set detection time equal to time when COLREGS start applying
        # OBS! Changing this will effect initial conditions in several functions

        self.epsilon_course = np.deg2rad(epsilon_course)
        self.epsilon_speed = epsilon_speed
        self.phi_OT_min = np.deg2rad(phi_OT_min)  # Minimum relative bearing defining an overtaking encounter
        self.phi_OT_max = np.deg2rad(phi_OT_max)  # Maximum relative bearing defining an overtaking encounter
        # self.phi_SB_lim = np.deg2rad(phi_SB_lim)  # Defines a starboard turn in rule 14

        # alpha variables are relative bearing of own ship as seen from obstacle
        self.alpha_crit_13 = np.deg2rad(alpha_critical_13)
        self.alpha_crit_14 = np.deg2rad(alpha_critical_14)
        self.alpha_crit_15 = np.deg2rad(alpha_critical_15)

        # Book-keeping
        self.situation_matrix = np.zeros([self.n_vessels, self.n_vessels, self.vessels[0].n_msgs], dtype=int)
        self.cpa_idx = np.zeros([self.n_vessels, self.n_vessels], dtype=int)
        self.detection_idx = np.zeros([self.n_vessels, self.n_vessels], dtype=int)
        self.ranges = np.zeros([self.n_vessels, self.n_vessels, self.n_msgs], dtype=float)
        self.ranges_set = False
        self.delta_course_max = np.zeros([self.n_vessels, self.n_vessels], dtype=float)
        self.delta_speed_max = np.zeros([self.n_vessels, self.n_vessels], dtype=float)
        self.delta_speed_max_red = np.zeros([self.n_vessels, self.n_vessels], dtype=float)  # For Woerner's R17
        self.alpha_cpa = np.zeros([self.n_vessels, self.n_vessels], dtype=float)
        self.beta_cpa = np.zeros([self.n_vessels, self.n_vessels], dtype=float)
        self.relative_heading_set = False

        # New attributes
        self.alpha = np.zeros([self.n_vessels, self.n_vessels, self.n_msgs])
        self.beta = np.zeros([self.n_vessels, self.n_vessels, self.n_msgs])
        self.beta_180 = np.zeros([self.n_vessels, self.n_vessels, self.n_msgs])
        

    # Functions for determining applicable rules -----------------------------------------------------------------------
    def determine_situations(self, vessel):
        """Determine applicable rules the given vessel with regards to all other vessels.
        The situation is saved to the situation_matrix for each timestep. The situation at one given timestep is
        dependent on the situation in previous timesteps.
        """

        for obst in self.vessels:
            if vessel.id == obst.id:
                # The elements in the situation matrix are initialized to zero, i.e. no applicable rule.
                continue
            for i in range(self.n_msgs):
                # Overtaking situation
                if self.ranges[vessel.id, obst.id, i] > self.r_colregs[0]:  #r_colregs_2_max
                    # If outside COLREGS range
                    if abs(self.situation_matrix[vessel.id, obst.id, i - 1]) == self.OTSO:
                        # Overtaking situation passed when vessels are out of range
                        self.situation_matrix[vessel.id, obst.id, i] = self.OP
                    else:
                        # No applicable rules
                        # print(f"Vessel: {vessel.id}, Obstacle: {obst.id}, Time: {vessel.time_stamps[i]:.2f}")
                        # temp_in = input("Press enter to continue")
                        self.situation_matrix[vessel.id, obst.id, i] = self.NAR

                # Crossing situation or head-on
                elif self.ranges[vessel.id, obst.id, i] <= self.r_colregs[0]: #r_colregs_2_max
                    # If inside COLREGS stage 2, 3 or 4
                    obst_passed, os_passed = self.determine_applicable_rules(vessel, obst, i)
                    if abs(self.situation_matrix[vessel.id, obst.id, i - 1]) == self.CRSO \
                            or self.situation_matrix[vessel.id, obst.id, i - 1] == self.HO:
                        # If crossing or head-on at previous time step
                        if obst_passed and os_passed:
                            self.situation_matrix[vessel.id, obst.id, i] = self.OP  # Mark as passed
                        else:
                            self.situation_matrix[vessel.id, obst.id, i] = \
                                self.situation_matrix[vessel.id, obst.id, i - 1]  # Keep situation
                            
                    elif self.situation_matrix[vessel.id, obst.id, i - 1] == self.OTSO:
                        self.situation_matrix[vessel.id, obst.id, i] = \
                            self.situation_matrix[vessel.id, obst.id, i - 1]  # Keep situation
                        
                    elif self.situation_matrix[vessel.id, obst.id, i - 1] == self.OTGW:
                        self.situation_matrix[vessel.id, obst.id, i] = \
                            self.situation_matrix[vessel.id, obst.id, i - 1]  # Keep situation
                    if obst_passed and os_passed:
                        self.situation_matrix[vessel.id, obst.id, i] = self.OP

    def determine_applicable_rules(self, vessel, obst, i):
        """
        Determine applicable COLREGS rules at a given sample index i. Note that this function does not take the historic
        track data or velocities into account.
        :param vessel:
        :param obst:
        :param i: Index of sample time where check is to be done
        """
        print(f"Current angle of vessel {vessel.id}: {np.rad2deg(vessel.state[2, i]):.2f}, at time: {vessel.time_stamps[i]:.2f}")
        dist_to_obst = np.empty(2)
        dist_to_obst[0] = obst.state[0, i] - vessel.state[0, i]
        dist_to_obst[1] = obst.state[1, i] - vessel.state[1, i]
       
        # Relative bearing of obstacle as seen from own ship
        beta = normalize_2pi(normalize_2pi(np.arctan2(dist_to_obst[0], dist_to_obst[1])) - vessel.state[2, i])
        beta_180 = normalize_pi(beta)
        # Relative bearing of own ship as seen from the obstacle
        alpha = normalize_pi(normalize_2pi(np.arctan2(-dist_to_obst[0], -dist_to_obst[1])) - obst.state[2, i])
        alpha_360 = normalize_2pi(alpha)
        
        # with open("angles.txt", "a") as f:
        #     f.write(f"{vessel.id}, {obst.id}, {i}, {np.rad2deg(beta)}, {np.rad2deg(beta_180)}, {np.rad2deg(alpha)}, {np.rad2deg(alpha_360)}\n")

        # Check for overtaking
        if (self.phi_OT_min < beta < self.phi_OT_max) and (abs(alpha) < self.alpha_crit_13) \
                and (vessel.speed[i] < obst.speed[i]):
            # Own-ship is being overtaken by obstacle j and is the stand on vessel.
            self.situation_matrix[vessel.id, obst.id, i] = self.OTSO

        elif (self.phi_OT_min < alpha_360 < self.phi_OT_max) and (abs(beta_180) < self.alpha_crit_13)\
                 and (vessel.speed[i] > obst.speed[i]):
            # Own-ship is overtaking obstacle j and is the give way vessel.
            self.situation_matrix[vessel.id, obst.id, i] = self.OTGW

        # Check for head-on
        elif (abs(beta_180) < self.alpha_crit_14) and (abs(alpha) < self.alpha_crit_14):
            # Head on situation
            self.situation_matrix[vessel.id, obst.id, i] = self.HO

        # Check for crossing
        elif (beta < self.phi_OT_min) and (-self.phi_OT_min < alpha < self.alpha_crit_15):
            # Crossing situation, own-ship is give-way vessel
            self.situation_matrix[vessel.id, obst.id, i] = self.CRGW
        
        elif (alpha_360 < self.phi_OT_min) and (-self.phi_OT_min < beta_180 < self.alpha_crit_15):
            # Crossing situation, own-ship is stand-on vessel
            self.situation_matrix[vessel.id, obst.id, i] = self.CRSO
        
        # No applicable rules at current time
        else:
            self.situation_matrix[vessel.id, obst.id, i] = self.NAR

        obst_passed = False
        os_passed = False
        # Removed since the velocities already are in NED
        # vo = rotate(obst.state[3:5, i], obst.state[2, i])
        # vs = rotate(vessel.state[3:5, i], vessel.state[2, i])
        vo = obst.state[3:5, i]
        vs = vessel.state[3:5, i]
        los = dist_to_obst / np.linalg.norm(dist_to_obst)
        if np.dot(vo, -los) < np.cos(self.phi_OT_min) * np.linalg.norm(vo):
            os_passed = True    # Own ship passed obstacle

        if np.dot(vs, los) < np.cos(self.phi_OT_min) * np.linalg.norm(vs):
            obst_passed = True  # Obstacle passed own ship

        return obst_passed, os_passed

    def entry_criteria(self, vessel, obst, i):
        """
        Determine applicable COLREGS rules at a given sample index i. Note that this function does not take the historic
        track data or velocities into account.
        :param vessel:
        :param obst:
        :param i: Index of sample time where check is to be done
        """
        cpa_indx = self.cpa_idx[vessel.id, obst.id]

        if vessel.not_moving \
                or np.isnan(vessel.state[2, i]) \
                or np.isnan(obst.state[2, i]) \
                or self.ranges[vessel.id, obst.id, cpa_indx] > 5000:
            self.situation_matrix[vessel.id, obst.id, i] = self.NAR
            return self.situation_matrix[vessel.id, obst.id, i]

        # todo: Add speed vector check for overtaking.
        dist_to_obst = np.empty(2)
        dist_to_obst[0] = obst.state[0, i] - vessel.state[0, i]
        dist_to_obst[1] = obst.state[1, i] - vessel.state[1, i]
        tot_dist = np.linalg.norm(dist_to_obst)

        # Todo: check the conditionals below.
        # # Relative bearing of obstacle as seen from own ship
        beta = normalize_2pi(normalize_2pi(np.arctan2(dist_to_obst[1], dist_to_obst[0])) - vessel.state[2, i])
        beta_180 = normalize_pi(beta)
        # Relative bearing of own ship as seen from the obstacle
        if obst.speed[i] < 0.2:
            alpha = np.pi
            alpha_360 = normalize_2pi(alpha)
        else:
            alpha = normalize_pi(normalize_2pi(np.arctan2(-dist_to_obst[1], -dist_to_obst[0])) - obst.state[2, i])
            alpha_360 = normalize_2pi(alpha)

        self.alpha[vessel.id, obst.id, i] = alpha
        self.beta[vessel.id, obst.id, i] = beta
        self.beta_180[vessel.id, obst.id, i] = beta_180

        if i == 0:
            try:
                nan_indx = np.append(np.array([i for i, x in enumerate(vessel.speed) if np.isnan(x)]),
                                     [0, len(vessel.speed)])
                if vessel.speed[np.isnan(vessel.speed) == False].max() < 0.5 \
                        or cpa_indx in nan_indx - 1 or cpa_indx in nan_indx + 1 \
                        or vessel.travel_dist < 1000:
                    vessel.not_moving = True
                    return self.situation_matrix[vessel.id, obst.id, i]
            except:
                vessel.not_moving = True
                return self.situation_matrix[vessel.id, obst.id, i]

        if vessel.not_moving == True:
            return self.situation_matrix[vessel.id, obst.id, i]

        if self.ranges[vessel.id, obst.id, i] > 15000 \
                or vessel.speed[i] < 0.1:
            # No applicable rules at current time
            self.situation_matrix[vessel.id, obst.id, i] = self.NAR
            return self.situation_matrix[vessel.id, obst.id, i]

        if (beta > self.phi_OT_min) and (beta < self.phi_OT_max) and (abs(alpha) < self.alpha_crit_13) \
                and (vessel.speed[i] < obst.speed[i] + 0.2):
            # Own-ship is being overtaken by obstacle j and is the stand on vessel.
            self.situation_matrix[vessel.id, obst.id, i] = self.OTSO

        elif (self.phi_OT_min < alpha_360 < self.phi_OT_max) \
                and (abs(beta_180) < self.alpha_crit_13) and (vessel.speed[i] > obst.speed[i] + 0.2):
            # Own-ship is overtaking obstacle j and is the give way vessel.
            self.situation_matrix[vessel.id, obst.id, i] = self.OTGW

        elif (abs(beta_180) < self.alpha_crit_14) and (abs(alpha) < self.alpha_crit_14):
            # Head on situation
            self.situation_matrix[vessel.id, obst.id, i] = self.HO

        elif (alpha_360 < self.phi_OT_min) and (beta_180 > -self.phi_OT_min) and (beta_180 < self.alpha_crit_15):
            # Crossing situation, own-ship is give-way vessel
            self.situation_matrix[vessel.id, obst.id, i] = self.CRGW

        elif (beta < self.phi_OT_min) and (alpha > -self.phi_OT_min) and (alpha < self.alpha_crit_15):
            # Crossing situation, own-ship is stand-on vessel
            self.situation_matrix[vessel.id, obst.id, i] = self.CRSO

        else:
            # No applicable rules at current time
            self.situation_matrix[vessel.id, obst.id, i] = self.NAR

        # if (self.using_coastline) and (self.situation_matrix[vessel.id, obst.id, i] != self.NAR):
        #    x_width = abs(vessel.stateLonLat[0,i] - obst.stateLonLat[0,i])
        #    x_width = max(0.01, x_width)
        #    y_width = abs(vessel.stateLonLat[1,i] - obst.stateLonLat[1,i])
        #    y_width = max(0.01, y_width)
        #
        #    coast_inside = self.coastline.loc[(self.coastline['lon'] > (max(vessel.stateLonLat[0, i],
        #                                                                    obst.stateLonLat[0, i]) - x_width)) &\
        #                                      (self.coastline['lon'] < (min(vessel.stateLonLat[0, i],
        #                                                                    obst.stateLonLat[0,i]) + x_width)) &\
        #                                      (self.coastline['lat'] > (max(vessel.stateLonLat[1, i],
        #                                                                    obst.stateLonLat[1,i]) - y_width)) &\
        #                                      (self.coastline['lat'] < \
        #                                       (min(vessel.stateLonLat[1, i],
        #                                            obst.stateLonLat[1,i]) + y_width))].reset_index(drop = True)

          # if len(coast_inside.index) != 0:
          #     vector_1 = np.array([vessel.stateLonLat[0,i] - obst.stateLonLat[0,i],
          #                          vessel.stateLonLat[1,i] - obst.stateLonLat[1,i]])
          #     for lon, lat in zip(coast_inside.lon.tolist(), coast_inside.lat.tolist()):
          #         vector_2 = np.array([vessel.stateLonLat[0,i] - lon, vessel.stateLonLat[1,i] - lat])
          #         unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
          #         unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
          #         dot_product = np.dot(unit_vector_1, unit_vector_2)
        #
        #           angle_OVK = np.arccos(dot_product)
        #
        #           if abs(np.linalg.norm(np.sin(angle_OVK)*vector_2) < 0.001):
        #               vector_2 = -vector_2 + vector_1
        #
        #               unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        #               unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        #               dot_product = np.dot(unit_vector_1, unit_vector_2)
        #
        #               angle_KOV = normalize_pi(np.arccos(dot_product))
        #               if abs(angle_KOV) > np.deg2rad(45):
        #                   continue
        #
        #
        #               if False: # print vector to land
        #                   print("\n\n\n\n"+ str(lon) + " -- " + str(lat))
        #                   print(vessel.stateLonLat[:,i])
        #                   print(obst.stateLonLat[:,i])
        #                   print("angle_KOV:" + str(angle_KOV))
        #                   print("angle_OVK:" + str(angle_OVK))
        #                   print(vector_1)
        #                   print(vector_2)
        #                   print(np.linalg.norm(np.sin(angle_OVK)*vector_2))
        #
        #               self.situation_matrix[vessel.id, obst.id, i] = self.NAR

        return self.situation_matrix[vessel.id, obst.id, i]

    def filter_out_non_complete_situations(self, vessel, obst):

        debug = False

        if debug:
            print(vessel.name, " - ", obst.name)

        if self.cpa_idx[vessel.id, obst.id] == 0 or \
                self.cpa_idx[vessel.id, obst.id] >= np.where(np.isnan(vessel.speed) == False)[-1][-1]:
            self.situation_matrix[vessel.id, obst.id, :] = self.NAR
            return self.situation_matrix[vessel.id, obst.id, :]

        # Filter Head on situations
        if self.HO in self.situation_matrix[vessel.id, obst.id, :]:
            start = np.argmax(self.situation_matrix[vessel.id, obst.id, :] == self.HO)  # start of sit
            obst_yaw_start = obst.state[2, start]
            end = start + np.argmax(
                np.array([abs_ang_diff(a, obst_yaw_start) for a in obst.state[2, start:]]) > 0.5 * np.pi)
            end = end if end != start else self.n_msgs

            alpha_abs = abs(self.alpha[vessel.id, obst.id, start:end + 1])
            alpha_max = max(alpha_abs)
            if alpha_max < 0.5 * np.pi:
                area = self.situation_matrix[vessel.id, obst.id] == self.HO
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR
            else:
                beta_max = max(abs(self.beta_180[vessel.id, obst.id, start:end + 1]))

                if beta_max < 0.5 * np.pi:
                    area = self.situation_matrix[vessel.id, obst.id, :] == self.HO
                    self.situation_matrix[vessel.id, obst.id, area] = self.NAR

        # Filter Overtake stay on situations
        if self.OTSO in self.situation_matrix[vessel.id, obst.id, :]:
            # Start index of situation
            start = np.argmax(self.situation_matrix[vessel.id, obst.id, :] == self.OTSO)

            # The end index of situation is set to either when the obstacle ship have turned 
            # 45 degrees from the initial yaw or the end of case if the difference is large enough
            obst_yaw_start = obst.state[2, start]
            end = start + np.argmax(
                np.array([abs_ang_diff(a, obst_yaw_start) for a in obst.state[2, start:]]) > 0.25 * np.pi)
            end = end if end != start else self.n_msgs

            if debug:
                print("Situation (OT) start stop:", start, "--", end)

            alpha_abs = abs(self.alpha[vessel.id, obst.id, start:end + 1])
            alpha_min = min(alpha_abs)

            if alpha_min > 0.5 * np.pi or self.cpa_idx[vessel.id, obst.id] < start \
                    or self.cpa_idx[vessel.id, obst.id] > end:
                area = self.situation_matrix[vessel.id, obst.id, :] == self.OTSO
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR
            else:
                beta_abs = abs(self.beta_180[vessel.id, obst.id, start:end + 1])
                beta_max = max(beta_abs)

                if beta_max < 0.5 * np.pi:
                    area = self.situation_matrix[vessel.id, obst.id, :] == self.OTSO
                    self.situation_matrix[vessel.id, obst.id, area] = self.NAR

            # CPA Criteria
            alpha_cpa_abs = abs(self.alpha[vessel.id, obst.id, self.cpa_idx[vessel.id, obst.id]])
            beta_cpa_abs = abs(self.beta_180[vessel.id, obst.id, self.cpa_idx[vessel.id, obst.id]])

            if debug:
                print("Alpha_cpa: ", alpha_cpa_abs, ", beta_cpa_180: ", beta_cpa_abs)

            if alpha_cpa_abs < np.pi / 6 or alpha_cpa_abs > 5 * np.pi / 6:
                area = self.situation_matrix[vessel.id, obst.id, :] == self.OTSO
                if debug:
                    print(area)
                    print(self.situation_matrix[vessel.id, obst.id, :])
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR
                if debug:
                    print(area)
                    print(self.situation_matrix[vessel.id, obst.id, :])

            if beta_cpa_abs < np.pi / 6 or beta_cpa_abs > 5 * np.pi / 6:
                area = self.situation_matrix[vessel.id, obst.id, :] == self.OTSO
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR

        # Filter Overtake give way situations
        if self.OTGW in self.situation_matrix[vessel.id, obst.id, :]:
            start = np.argmax(self.situation_matrix[vessel.id, obst.id, :] == self.OTGW)  # start of sit
            obst_yaw_start = obst.state[2, start]

            end = start + np.argmax(
                np.array([abs_ang_diff(a, obst_yaw_start) for a in obst.state[2, start:]]) > 0.5 * np.pi)
            end = end if end != start else self.n_msgs

            alpha_abs = abs(self.alpha[vessel.id, obst.id, start:end + 1])
            alpha_max = max(alpha_abs)

            if alpha_max < 0.5 * np.pi or self.cpa_idx[vessel.id, obst.id] < start \
                    or self.cpa_idx[vessel.id, obst.id] > end:
                area = self.situation_matrix[vessel.id, obst.id, :] == self.OTGW
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR
            else:
                beta_abs = abs(self.beta_180[vessel.id, obst.id, start:end + 1])
                beta_min = min(beta_abs)

                if beta_min > 0.5 * np.pi:
                    area = self.situation_matrix[vessel.id, obst.id, :] == self.OTGW
                    self.situation_matrix[vessel.id, obst.id, area] = self.NAR

            # CPA Criteria
            alpha_cpa_abs = abs(self.alpha[vessel.id, obst.id, self.cpa_idx[vessel.id, obst.id]])
            beta_cpa_abs = abs(self.beta_180[vessel.id, obst.id, self.cpa_idx[vessel.id, obst.id]])

            if debug:
                print("BEFORE, alpha:", alpha_cpa_abs, ", beta:", beta_cpa_abs)
                print(self.situation_matrix[vessel.id, obst.id, :])

            if alpha_cpa_abs < np.pi / 6 or alpha_cpa_abs > 5 * np.pi / 6:
                area = self.situation_matrix[vessel.id, obst.id, :] == self.OTGW
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR

            if beta_cpa_abs < np.pi / 6 or beta_cpa_abs > 5 * np.pi / 6:
                area = self.situation_matrix[vessel.id, obst.id, :] == self.OTGW
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR

            if debug:
                print("AFTER")
                print(self.situation_matrix[vessel.id, obst.id, :])

        # CROSSING
        if (self.CRGW or self.CRSO) in self.situation_matrix[vessel.id, obst.id, :]:
            from scipy.spatial.distance import cdist

            # Check that both vessels move more than 100 meters
            xa = np.array(vessel.state[0:2, :]).transpose()
            xb = np.array(obst.state[0:2, :]).transpose()

            cdist = np.min(cdist(xa, xb, metric='euclidean'))

            if cdist > 100:
                area = (self.situation_matrix[vessel.id, obst.id, :] == self.CRGW) | (
                        self.situation_matrix[vessel.id, obst.id, :] == self.CRSO)
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR

            # Check for at least one vessel crosses the other's LOS
            sit = ((self.situation_matrix[vessel.id, obst.id, :] == self.CRGW)
                   | (self.situation_matrix[vessel.id, obst.id, :] == self.CRSO))
            start = np.argmax(sit)  # start of sit
            alpha_sign_change = np.where(np.sign(self.alpha[vessel.id, obst.id, start:-1]) != np.sign(
                self.alpha[vessel.id, obst.id, start + 1:]))[0] + 1 + start
            beta_sign_change = np.where(np.sign(self.beta_180[vessel.id, obst.id, start:-1]) != np.sign(
                self.beta_180[vessel.id, obst.id, start + 1:]))[0] + 1 + start
            if len(alpha_sign_change) == 0 and len(beta_sign_change) == 0:
                area = (self.situation_matrix[vessel.id, obst.id, :] == self.CRGW) | (
                        self.situation_matrix[vessel.id, obst.id, :] == self.CRSO)
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR

            if False:  # DEBUG
                print("cdist:", cdist)
                print(self.situation_matrix[vessel.id, obst.id, :])

        if debug:
            print(self.situation_matrix[vessel.id, obst.id, :])

        return self.situation_matrix[vessel.id, obst.id, :]

    # Functions for calculating necessary parameters -------------------------------------------------------------------
    def find_ranges(self):
        """
        Calculate ranges between all vessels. Also finds index of obstacle detection and CPA.
        Sets the ranges_set parameter to True.
        """
        for vessel in self.vessels:
            for obst in self.vessels:
                if vessel.id < obst.id:
                    for i in range(self.n_msgs):
                        self.ranges[(vessel.id, obst.id), (obst.id, vessel.id), i] = \
                            np.linalg.norm(vessel.state[0:2, i] - obst.state[0:2, i])
                    self.detection_idx[(vessel.id, obst.id), (obst.id, vessel.id)] = \
                        np.argmax(self.ranges[vessel.id, obst.id] <= self.r_detect)
                    r_cpa = np.min(self.ranges[vessel.id, obst.id])
                    self.cpa_idx[(vessel.id, obst.id), (obst.id, vessel.id)] = \
                        np.argmax(self.ranges[vessel.id, obst.id] == r_cpa) 
        self.ranges_set = True

class Vessel:
    # Trajectories are defined in the ENU frame
    state = None  # [ x y yaw u v ] [m m rad m/s m/s]
    maneuver_der = None  # [first_derivative second_derivative]

    def __init__(self, name, n_msgs):
        self.n_msgs = n_msgs
        self.name = name
        self.callsign = None
        self.mmsi = None
        self.imo = None
        self.not_moving = False  # used to quicker evaluate COLREG
        self.state = np.empty([5, n_msgs])  # [x y yaw u v]
        self.stateLonLat = np.empty([2, n_msgs])  # only to integrate coastline
        self.stateDateTime = np.empty(n_msgs)  # Only to carry on to Parameters
        self.debug_state = np.empty([5, n_msgs])
        self.maneuver_der = np.empty([3, n_msgs])  # derivatives of yaw
        self.lat = np.zeros(n_msgs)
        self.lon = np.zeros(n_msgs)
        self.speed = np.empty(n_msgs)
        self.speed_der = np.empty(n_msgs)
        self.true_heading = np.empty(n_msgs)
        self.message_nr = np.zeros(n_msgs)
        self.nav_status = np.zeros(n_msgs)
        self.nan_idx = np.zeros([2, 1], dtype=int)
        self.travel_dist = None
        self.msgs_idx = np.empty(n_msgs, dtype=int)
        self.time_stamps = np.empty(n_msgs, dtype=float)
        self.dT = None
        self.id = None
        self.length = 0
        self.width = 0
        self.type = 0

        self.maneuver_detect_idx = None
        self.maneuver_start_stop = None
        self.delta_course = None
        self.delta_speed = None
        self.delta_course_signed = None
        self.delta_speed_signed = None
        self.maneuvers_found = False
        self.maneuvers_searched = False

        self.colreg_maneuver_idx = []
        self.colreg_maneuvers_made = []
        self.maneuver_range = []
        self.maneuver_delta_course = []
        self.maneuver_delta_speed = []

    def plot_trajectory(self, ax):
        out = ax.plot(self.state[0], self.state[1], label=self.name)
        return out

    class Maneuver:
        # Subclass containing data for each maneuver made by vessel

        def __init__(self, maneuver_idx, delta_course, delta_speed):
            self.idx = maneuver_idx
            self.course = delta_course
            self.speed = delta_speed


@dataclass
class Parameters:
    n_ships: int
    own_mmsi: int
    obst_mmsi: int
    own_name: str
    obst_name: str
    own_callsign: str
    obst_callsign: str
    own_length: float
    obst_length: float
    own_width: float
    obst_width: float
    own_type: int
    obst_type: int
    own_nav_status: float
    obst_nav_status: float
    own_speed: float
    obst_speed: float
    multi_man_own: bool
    maneuver_made_own: bool
    maneuver_index_own: int
    maneuver_stop_idx_own: int
    r_maneuver_own: float
    pre_man_dist_own: float
    pre_man_t_cpa_own: int
    post_man_dist_own: float
    post_man_t_cpa_own: int
    delta_speed_own: float
    delta_course_own: float
    multi_man_obst: bool
    maneuver_made_obst: bool
    maneuver_index_obst: int
    maneuver_stop_idx_obst: int
    r_maneuver_obst: float
    pre_man_dist_obst: float
    pre_man_t_cpa_obst: int
    post_man_dist_obst: float
    post_man_t_cpa_obst: int
    delta_speed_obst: float
    delta_course_obst: float
    alpha_start: float
    beta_start: float
    r_cpa: float
    alpha_cpa: float
    beta_cpa: float
    lon_maneuver: float
    lat_maneuver: float
    COLREG: float
    single_COLREG_type: bool
    time: datetime.time
    date_cpa: datetime
    cpa_idx: int
    start_idx: int
    stop_idx: int

def start_plot(ax):
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