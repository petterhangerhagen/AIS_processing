import numpy as np
import pandas as pd
import datetime
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import warnings
from AV_class_functions.helper_methods import *

from matplotlib import cm
from dataclasses import dataclass

warnings.filterwarnings("ignore")


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
        # Coastline incorporation

        # try:
        #    self.coastline = pd.read_csv("coastline.csv", sep = ';')
        #    self.using_coastline = True
        # except:
        #    #print("Could not read coastline.csv")
        #    self.using_coastline = False 

        self.vessels = vessels
        self.n_vessels = len(vessels)
        self.n_msgs = self.vessels[0].n_msgs
        # # # if len(ais_path) != 0:
        # # #     self.read_AIS(ais_path, ship_path)

        # # # # Just a slightly convoluted method of storing case name/code
        # # # case_name = ais_path.replace("-sec.csv", "")
        # # # self.case_name = case_name
        # # # for i in reversed(range(len(case_name))):
        # # #     if ais_path[i] == "-":
        # # #         self.case_name = case_name.replace(case_name[i - len(case_name):], "")[-5:]
        # # #         break

        # # # self.n_vessels = len(self.vessels)
        # # # if self.n_vessels == 0:
        # # #     print("No vessels in file:", ais_path)
        # # #     return

        # # # self.n_msgs = self.vessels[0].n_msgs
        # # # for vessel in self.vessels:
        # # #     self.n_msgs = min(self.n_msgs, vessel.n_msgs)

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

        # self.angles = np.zeros([self.n_vessels, self.n_msgs, 4])
        with open("angles.txt", "w") as f:
            f.write("")

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

        dist_to_obst = np.empty(2)
        dist_to_obst[0] = obst.state[0, i] - vessel.state[0, i]
        dist_to_obst[1] = obst.state[1, i] - vessel.state[1, i]
       
        # Relative bearing of obstacle as seen from own ship
        beta = normalize_2pi(normalize_2pi(np.arctan2(dist_to_obst[1], dist_to_obst[0])) - vessel.state[2, i])
        beta_180 = normalize_pi(beta)
        # Relative bearing of own ship as seen from the obstacle
        alpha = normalize_pi(normalize_2pi(np.arctan2(-dist_to_obst[1], -dist_to_obst[0])) - obst.state[2, i])
        alpha_360 = normalize_2pi(alpha)
        
        with open("angles.txt", "a") as f:
            f.write(f"{vessel.id}, {obst.id}, {i}, {np.rad2deg(beta)}, {np.rad2deg(beta_180)}, {np.rad2deg(alpha)}, {np.rad2deg(alpha_360)}\n")
        
        # Check for overtaking
        if (beta > self.phi_OT_min) and (beta < self.phi_OT_max) and (abs(alpha) < self.alpha_crit_13) \
                and (vessel.speed[i] < obst.speed[i]):
            # Own-ship is being overtaken by obstacle j and is the stand on vessel.
            self.situation_matrix[vessel.id, obst.id, i] = self.OTSO
        elif (alpha_360 > self.phi_OT_min) and (alpha_360 < self.phi_OT_max) \
                and (abs(beta_180) < self.alpha_crit_13) and (vessel.speed[i] > obst.speed[i]):
            # Own-ship is overtaking obstacle j and is the give way vessel.
            self.situation_matrix[vessel.id, obst.id, i] = self.OTGW

        # Check for head-on
        elif (abs(beta_180) < self.alpha_crit_14) and (abs(alpha) < self.alpha_crit_14):
            # Head on situation
            self.situation_matrix[vessel.id, obst.id, i] = self.HO

        # Check for crossing
        elif (alpha_360 < self.phi_OT_min) and (beta_180 > -self.phi_OT_min) and (beta_180 < self.alpha_crit_15):
            # Crossing situation, own-ship is give-way vessel
            self.situation_matrix[vessel.id, obst.id, i] = self.CRGW
        elif (beta < self.phi_OT_min) and (alpha > -self.phi_OT_min) and (alpha < self.alpha_crit_15):
            # Crossing situation, own-ship is stand-on vessel
            self.situation_matrix[vessel.id, obst.id, i] = self.CRSO
        
        # No applicable rules at current time
        else:
            # No applicable rules at current time
            self.situation_matrix[vessel.id, obst.id, i] = self.NAR

        obst_passed = False
        os_passed = False
        vo = rotate(obst.state[3:5, i], obst.state[2, i])
        vs = rotate(vessel.state[3:5, i], vessel.state[2, i])
        los = dist_to_obst / np.linalg.norm(dist_to_obst)
        if np.dot(vo, -los) < np.cos(self.phi_OT_min) * np.linalg.norm(vo):
            os_passed = True

        if np.dot(vs, los) < np.cos(self.phi_OT_min) * np.linalg.norm(vs):
            obst_passed = True
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

    def find_relative_heading(self):
        """
        Find the relative bearings between the vessels (alpha and beta values) at CPA. The row and column number
        signifies the own ship and obstacle vessel index respectively.
        Sets the relative_bearing_set parameter to True.
        """
        for vessel in self.vessels:
            for obst in self.vessels:
                if vessel.id < obst.id:
                    cpa_idx = self.cpa_idx[vessel.id, obst.id]
                    dist_to_obst = np.empty(2)
                    dist_to_obst[0] = obst.state[0, cpa_idx] - vessel.state[0, cpa_idx]
                    dist_to_obst[1] = obst.state[1, cpa_idx] - vessel.state[1, cpa_idx]

                    # Viewed from vessel
                    self.alpha_cpa[vessel.id, obst.id] = normalize_pi(
                        normalize_2pi(np.arctan2(-dist_to_obst[1], -dist_to_obst[0])) - obst.state[2, cpa_idx])  # alpha
                    self.beta_cpa[vessel.id, obst.id] = normalize_2pi(
                        normalize_2pi(np.arctan2(dist_to_obst[1], dist_to_obst[0])) - vessel.state[2, cpa_idx])  # beta

                    # Viewed from obst
                    self.alpha_cpa[obst.id, vessel.id] = normalize_pi(
                        normalize_2pi(np.arctan2(dist_to_obst[1], dist_to_obst[0])) - vessel.state[2, cpa_idx])  # alpha
                    self.beta_cpa[obst.id, vessel.id] = normalize_2pi(
                        normalize_2pi(np.arctan2(-dist_to_obst[1], -dist_to_obst[0])) - obst.state[2, cpa_idx])  # beta

        self.relative_heading_set = True

    def find_maneuver_detect_index(self, vessel):
        """
        Find indices i where the vessel's speed and/or course change exceeds epsilon_speed and/or epsilon_course
        respectively. The change is defined as the difference between the speed/course at index i and
        index i + step_length, where the step length is defined by the sample frequency of the own_ship's state such
        that the time between sample i and i + step_length is one second.
        """
        # TODO: Set limits as class parameters instead of hardcoded values.
        # TODO: Split maneuver if course change changes direction.
        if vessel.maneuvers_searched:
            return

        if vessel.travel_dist < 1000:    #1000:
            vessel.maneuver_detect_idx = np.array([])
            # print(vessel.delta_course)
            vessel.delta_course = np.array([]) #vessel.delta_course([])
            vessel.delta_speed = []
            vessel.maneuvers_searched = True
            return

        vessel.maneuvers_searched = True  # Assure that computation is only done once

        step_length = 1
        i_maneuver_detect = np.array([])
        second_der_zeroes = np.array([])
        cont_man = False

        for i in range(vessel.n_msgs - step_length):
            if i > 0:
                if np.sign(vessel.maneuver_der[0, i]) != np.sign(vessel.maneuver_der[0, i - 1]) \
                        or np.sign(vessel.maneuver_der[2, i]) != np.sign(vessel.maneuver_der[2, i - 1]):
                    cont_man = False

            if np.abs(vessel.maneuver_der[0, i]) < 0.01:
                continue

            if np.abs(vessel.maneuver_der[1, i]) > 0.01 and np.sign(vessel.maneuver_der[1, i]) == np.sign(
                    vessel.maneuver_der[1, i - 1]):
                second_der_zeroes = np.concatenate([second_der_zeroes, [i]])
                continue

            if np.abs(vessel.maneuver_der[2, i]) < 0.005:
                continue

            if np.sign(vessel.maneuver_der[0, i]) == np.sign(vessel.maneuver_der[2, i]):
                continue

            if not cont_man:
                cont_man = True
                i_maneuver_detect = np.concatenate([i_maneuver_detect, [i]])
                second_der_zeroes = second_der_zeroes[-1:]

        i_maneuver_detect = [int(i) for i in i_maneuver_detect]

        speed_changes = [1 if vessel.speed_der[i] > self.epsilon_speed else 0 for i in range(len(vessel.speed_der))]
        speed_maneuvers = []
        start, stop = 0, 0
        in_man = False

        for i, v in enumerate(speed_changes):
            if not in_man:
                if v == 1:
                    start = i
                    in_man = True
            else:
                if v == 0:
                    stop = i - 1
                    in_man = False

                    speed_maneuvers.append([start, stop])

        delta_course_list = []
        delta_speed_list = []
        maneuver_idx_list = []
        maneuver_start_stop = []

        third_derivative_zeroes_bool = [
            np.sign(vessel.maneuver_der[2, i]) != np.sign(vessel.maneuver_der[2, i + 1]) or vessel.maneuver_der[
                2, i] == 0
            for i in range(len(vessel.maneuver_der[2, :]) - 1)]
        third_derivative_zeroes_bool = np.append(third_derivative_zeroes_bool, [False])

        third_derivative_zero_idx = np.array([i if b else 0 for i, b in enumerate(third_derivative_zeroes_bool)])
        third_derivative_zero_idx = third_derivative_zero_idx[third_derivative_zero_idx != 0]

        while len(i_maneuver_detect) > 0:
            i = i_maneuver_detect[0]

            above = third_derivative_zero_idx[third_derivative_zero_idx > i]
            below = third_derivative_zero_idx[third_derivative_zero_idx < i]

            val_above = int(above[0]) if len(above) > 0 else 0
            val_below = int(below[-1]) if len(below) > 0 else -1

            remove_course = True
            remove_speed = False

            if len(speed_maneuvers) > 0:
                if val_below > speed_maneuvers[0][1]:  # > start
                    if len(i_maneuver_detect) == 1:
                        remove_course = False
                        remove_speed = True
                        val_below = speed_maneuvers[0][0]
                        val_above = speed_maneuvers[0][1]
                        i = val_above
                else:
                    remove_speed = True
                    if val_above > speed_maneuvers[0][0]:
                        remove_course = False  # speed maneuver before course maneuver
                        val_below = speed_maneuvers[0][0]
                        val_above = speed_maneuvers[0][1]
                        i = val_above
                    else:
                        val_above = min(val_above, speed_maneuvers[0][0])
                        val_below = max(val_below, speed_maneuvers[0][1])

            delta_course_list.append(np.sum(vessel.maneuver_der[0, val_below:val_above + 1]))
            delta_speed_list.append(np.sum(vessel.speed_der[val_below:val_above + 1]))
            maneuver_idx_list.append(i)
            maneuver_start_stop.append([val_below, val_above + 1])

            if remove_course:
                i_maneuver_detect.pop(0)
            if remove_speed:
                speed_maneuvers.pop(0)

        i = 0
        remove = False
        mask = np.array([1] * len(maneuver_idx_list), dtype=bool)

        for item in maneuver_start_stop:
            if item in maneuver_start_stop[0:i]:
                remove = True
                mask[i] = False
            i += 1

        if remove:
            maneuver_idx_list = np.array(maneuver_idx_list)[mask]
            maneuver_start_stop = np.array(maneuver_start_stop)[mask]
            delta_course_list = np.array(delta_course_list)[mask]
            delta_speed_list = np.array(delta_speed_list)[mask]

        vessel.maneuver_detect_idx = np.array(maneuver_idx_list)
        vessel.maneuver_start_stop = np.array(maneuver_start_stop)
        vessel.delta_course = delta_course_list
        vessel.delta_speed = delta_speed_list

    def constructParams(self, own_vessel, obst_vessel, start_idx, stop_idx):
        """Takes the timespan of a COLREG situation and returns ownship info, obstacle info, and parameters for
        any maneuvers made by ownship. Output is returned as a Parameters-object.

        :param own_vessel: ownship as Vessel-object
        :param obst_vessel: obstacle as Vessel-object
        :param start_idx: Index for start of COLREG situaton
        :param stop_idx: Index for end of COLREG situaton
        """
        if not self.ranges_set:
            self.find_ranges()

        cpa_idx = self.cpa_idx[own_vessel.id, obst_vessel.id]
        if self.ranges[own_vessel.id, obst_vessel.id, cpa_idx] < 50:
            return None

        def getVesselParams(vessel, obst):
            if not vessel.maneuvers_found:
                printer_on = False  # Toggle printer

                self.find_maneuver_detect_index(vessel)

                if printer_on:
                    print("\n\n")
                    print("Maneuver [start stop]: ", vessel.maneuver_start_stop)
                    print("COLREGS start: ", start_idx, " stop: ", stop_idx)

                man_inside = ((start_idx < vessel.maneuver_start_stop[:, 0]) & (
                        stop_idx > vessel.maneuver_start_stop[:, 0])) if len(
                    vessel.maneuver_detect_idx) > 0 else np.array([], dtype=bool)
                man_inside = np.array(man_inside)

                if printer_on:
                    print("Maneuver inside COLREGS situation: ", man_inside)
                    print("Maneuver_detect_idx: ", vessel.maneuver_detect_idx[man_inside])

                maneuver_flag = True
                multi_man = False

                i = 0
                man_number = 0
                maneuver_idx = None
                maneuver_stop_idx = None

                pre_man_dist = None
                post_man_dist = None
                diff_man_dist = None

                pre_man_t_cpa = None
                post_man_t_cpa = None

                if len(vessel.maneuver_detect_idx[man_inside]) > 1:
                    multi_man = True
                    for inside in man_inside:
                        if not inside:
                            i += 1
                            continue

                        if printer_on:
                            print("Idx:", vessel.maneuver_detect_idx[i])
                            print("Man:", vessel.maneuver_start_stop[i])

                        range_val_start, time_to_cpa_start = calcPredictedCPA(vessel, obst,
                                                                              vessel.maneuver_start_stop[i][0])
                        range_val_stop, time_to_cpa_stop = calcPredictedCPA(vessel, obst,
                                                                            vessel.maneuver_start_stop[i][1])
                        if np.isnan(range_val_start) or np.isnan(range_val_stop):
                            continue
                        range_diff = range_val_stop - range_val_start
                        course_diff = signed_ang_diff(vessel.state[2, vessel.maneuver_start_stop[i][1]],
                                                      vessel.state[2, vessel.maneuver_start_stop[i][0]])

                        if printer_on:
                            print("R-start:", range_val_start)
                            print("R-stop:", range_val_stop)
                            print("R-diff:", range_diff)

                        if maneuver_idx is None:
                            if range_diff < 0:  # The maneuver decreases DCPA
                                continue
                        else:
                            if range_diff < diff_man_dist:  # Chose maneuver causing the largest increase in DCPA
                                continue

                        sit = self.situation_matrix[vessel.id, obst.id, start_idx]
                        if (sit == self.CRGW or sit == self.HO) and course_diff > 0:  # Port turn
                            continue

                        if printer_on:
                            print("NEW BEST")
                        maneuver_idx = vessel.maneuver_start_stop[i][0]
                        maneuver_stop_idx = vessel.maneuver_start_stop[i][1]
                        man_number = i
                        pre_man_dist, pre_man_t_cpa = range_val_start, time_to_cpa_start
                        post_man_dist, post_man_t_cpa = range_val_stop, time_to_cpa_stop
                        diff_man_dist = range_diff

                        i += 1

                    if printer_on:
                        print("FLAG")
                        input()

                elif len(vessel.maneuver_detect_idx[man_inside]) == 1:
                    for inside in man_inside:
                        if inside:
                            maneuver_idx = vessel.maneuver_start_stop[i][0]
                            maneuver_stop_idx = vessel.maneuver_start_stop[i][1]
                            man_number = i
                            pre_man_dist, pre_man_t_cpa = calcPredictedCPA(vessel, obst,
                                                                           vessel.maneuver_start_stop[i][0])
                            post_man_dist, post_man_t_cpa = calcPredictedCPA(vessel, obst,
                                                                             vessel.maneuver_start_stop[i][1])
                            # TODO: Add diff_man_dist or something similar that indicate evasive maneuver to parameters
                            #  e.g. Mark as evasive if post_man_t_cpa is zero.
                            if np.isnan(pre_man_dist) or np.isnan(post_man_dist):
                                break
                            diff_man_dist = post_man_dist - pre_man_dist
                            break
                        i += 1
                else:
                    maneuver_flag = False

                if maneuver_idx is None:
                    maneuver_flag = False

                return maneuver_flag, man_number, maneuver_idx, maneuver_stop_idx, multi_man, pre_man_dist, pre_man_t_cpa, post_man_dist, post_man_t_cpa

        maneuver_made_own, man_number_own, maneuver_idx_own, maneuver_stop_idx_own, multi_man_own, pre_man_dist_own, pre_man_t_cpa_own, post_man_dist_own, post_man_t_cpa_own = getVesselParams(
            own_vessel, obst_vessel)
        maneuver_made_obst, man_number_obst, maneuver_idx_obst, maneuver_stop_idx_obst, multi_man_obst, pre_man_dist_obst, pre_man_t_cpa_obst, post_man_dist_obst, post_man_t_cpa_obst = getVesselParams(
            obst_vessel, own_vessel)

        if maneuver_idx_own is None and maneuver_made_own:
            return None

        params = self.getParameters(
            own_vessel,
            obst_vessel,
            start_idx,
            stop_idx,
            maneuver_made_own,
            man_number_own,
            maneuver_idx_own,
            maneuver_stop_idx_own,
            multi_man_own,
            pre_man_dist_own,
            pre_man_t_cpa_own,
            post_man_dist_own,
            post_man_t_cpa_own,
            maneuver_made_obst,
            man_number_obst,
            maneuver_idx_obst,
            maneuver_stop_idx_obst,
            multi_man_obst,
            pre_man_dist_obst,
            pre_man_t_cpa_obst,
            post_man_dist_obst,
            post_man_t_cpa_obst)

        return params

    def getParameters(self, vessel, obst, start_idx, stop_idx, maneuver_made_own, man_number_own, maneuver_idx_own,
                      maneuver_stop_idx_own, multi_man_own, pre_man_dist_own, pre_man_t_cpa_own, post_man_dist_own,
                      post_man_t_cpa_own, maneuver_made_obst, man_number_obst, maneuver_idx_obst,
                      maneuver_stop_idx_obst, multi_man_obst, pre_man_dist_obst, pre_man_t_cpa_obst, post_man_dist_obst,
                      post_man_t_cpa_obst):
        """
        Takes the timespan of a COLREG situation and returns ownship info, obstacle info, and parameters for
        any maneuvers made by ownship. Output is returned as a Parameters-object.
        :param vessel: ownship as Vessel-object
        :param obst: obstacle as Vessel-object
        :param start_index: Index for start of COLREG situaton
        :param stop_index: Index for end of COLREG situaton
        :param maneuver_made_own: boolean True/False. True represents ownship having made a maneuver
        :param maneuver: Index for time at which maneuver takes place
        """
        if maneuver_made_own:
            idx = man_number_own
            delta_course_own = np.rad2deg(vessel.delta_course[man_number_own])
            delta_speed_own = vessel.delta_speed[man_number_own]
            r_maneuver_own = self.ranges[vessel.id, obst.id, maneuver_idx_own]
        else:
            maneuver_idx_own = (stop_idx - start_idx) // 2 + start_idx
            maneuver_stop_idx_own = maneuver_idx_own
            idx = maneuver_idx_own
            delta_course_own = 0
            delta_speed_own = 0
            r_maneuver_own = self.ranges[vessel.id, obst.id, idx]

        alpha_start = np.rad2deg(self.alpha[vessel.id, obst.id, start_idx])
        alpha_cpa = np.rad2deg(self.alpha[vessel.id, obst.id, self.cpa_idx[vessel.id, obst.id]])

        beta_start = np.rad2deg(self.beta[vessel.id, obst.id, start_idx])
        beta_cpa = np.rad2deg(self.beta[vessel.id, obst.id, self.cpa_idx[vessel.id, obst.id]])

        r_cpa = self.ranges[vessel.id, obst.id, self.cpa_idx[vessel.id, obst.id]]

        if r_cpa <= 50:
            return None

        lon_maneuver = vessel.stateLonLat[0, maneuver_idx_own]
        lat_maneuver = vessel.stateLonLat[1, maneuver_idx_own]

        own_speed = np.mean(vessel.speed[start_idx:stop_idx])
        obst_speed = np.mean(obst.speed[start_idx:stop_idx])

        colreg_type = self.situation_matrix[vessel.id, obst.id, start_idx:stop_idx].mean()
        single_colreg_type = np.all(self.situation_matrix[vessel.id, obst.id, start_idx:stop_idx] == colreg_type)

        time = convertSecondsToTime(int((stop_idx - start_idx) * (vessel.dT * 10 ** 9)))
        date_cpa = vessel.stateDateTime[self.cpa_idx[vessel.id, obst.id]]
        cpa_idx = self.cpa_idx[vessel.id, obst.id]

        # OBST param
        if maneuver_made_obst:
            idx = man_number_obst
            delta_course_obst = np.rad2deg(obst.delta_course[man_number_obst])
            delta_speed_obst = obst.delta_speed[man_number_obst]
            r_maneuver_obst = self.ranges[obst.id, vessel.id, maneuver_idx_obst]
        else:
            maneuver_idx_obst = (stop_idx - start_idx) // 2 + start_idx
            maneuver_stop_idx_obst = maneuver_idx_obst
            idx = maneuver_idx_obst
            delta_course_obst = 0
            delta_speed_obst = 0
            r_maneuver_obst = self.ranges[vessel.id, obst.id, idx]

        params = Parameters(len(self.vessels),
                            vessel.mmsi,
                            obst.mmsi,
                            vessel.name,
                            obst.name,
                            vessel.callsign,
                            obst.callsign,
                            vessel.length,
                            obst.length,
                            vessel.width,
                            obst.width,
                            vessel.type,
                            obst.type,
                            vessel.nav_status[start_idx:stop_idx + 1].mean(),
                            obst.nav_status[start_idx:stop_idx + 1].mean(),
                            own_speed,
                            obst_speed,
                            multi_man_own,
                            maneuver_made_own,
                            maneuver_idx_own,
                            maneuver_stop_idx_own,
                            r_maneuver_own,
                            pre_man_dist_own,
                            pre_man_t_cpa_own,
                            post_man_dist_own,
                            post_man_t_cpa_own,
                            delta_speed_own,
                            delta_course_own,
                            multi_man_own,
                            maneuver_made_obst,
                            maneuver_idx_obst,
                            maneuver_stop_idx_obst,
                            r_maneuver_obst,
                            pre_man_dist_obst,
                            pre_man_t_cpa_obst,
                            post_man_dist_obst,
                            post_man_t_cpa_obst,
                            delta_speed_obst,
                            delta_course_obst,
                            alpha_start,
                            beta_start,
                            r_cpa,
                            alpha_cpa,
                            beta_cpa,
                            lon_maneuver,
                            lat_maneuver,
                            colreg_type,
                            single_colreg_type,
                            time,
                            date_cpa,
                            cpa_idx,
                            start_idx,
                            stop_idx)
        return params


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
