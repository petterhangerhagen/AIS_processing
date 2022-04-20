import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import warnings

from matplotlib import cm
from dataclasses import dataclass, asdict

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
                 bag_path=[],
                 AIS_path=[],
                 ship_path=[],
                 r_colregs_2_max=7000,  # 5000
                 r_colregs_3_max=3000,
                 r_colregs_4_max=400,
                 r_pref=1500,  # TODO: Find the value used by Woerner
                 r_min=1000,
                 r_nm=800,
                 r_col=200,
                 s_r=0.5,  # TODO: Find the value used by Woerner
                 s_theta=0.5,  # TODO: Find the value used by Woerner
                 gamma_nm=0.4,
                 gamma_col=0.6,
                 epsilon_course=4,
                 epsilon_speed=0.5,
                 delta_chi_apparent=30,
                 delta_speed_apparent=5,
                 delta_chi_md=0,
                 delta_psi_md=2,
                 delta_speed_md=0.2,
                 delta_speed_reduction_apparent=0.5,
                 alpha_critical_13=45.0,  # 45,  # absolute value is used
                 alpha_critical_14=30.0,  # 13.0,  # absolute value is used
                 alpha_critical_15=0.0,  # -10.0
                 alpha_cpa_min_15=-25.0,  # -25
                 alpha_cpa_max_15=165.0,  # 165
                 alpha_ahead_lim_13=45.0,  # absolute value is used
                 alpha_cut=90,
                 beta_cut=90,
                 phi_OT_min=112.5,  # 112.5,  # equal
                 phi_OT_max=247.5,  # 247.5,  # equal
                 phi_SB_lim=-20.0,  # 20
                 gamma_ahead_13=0.3,
                 gamma_course_cpa_14=1,
                 gamma_delay_14=0.2,
                 gamma_nsb_14=0.6,
                 gamma_ahead_15=0.5,
                 gamma_17_safety=0.6,  # Value used by Minne
                 gamma_17_port_turn=0.7):  # Value used by Minne

        """
        :param bag_path: relative path to .bag file
        :type bag_path: sting
        :param AIS_path: relative path to .csv file containing encounter data
        :type AIS_path: sting
        :param ship_path: relative path to .csv file containing ship information
        :type ship_path: sting
        :param path: relative path to .bag or .csv file. Not currently in use
        :type path: sting
        :param r_colregs_2_max: [m] Maximum range for COLREGS stage 2.
        :type r_colregs_2_max: int
        :param r_colregs_3_max: [m] Maximum range for COLREGS stage 3.
        :type r_colregs_3_max: int
        :param r_colregs_4_max: [m] Maximum range for COLREGS stage 4. Usually four ship lengths.
        :type r_colregs_4_max: int
        :param r_pref: Preferred passing distance. [m]
        :type r_pref: int
        :param r_min: Minimum acceptable passing distance at CPA. [m]
        :type r_min: int
        :param r_nm:  Near miss passing distance. [m]
        :type r_nm: int
        :param r_col: Distance at CPA where a collision most likely would have occurred. [m]
        :type r_col: int
        :param gamma_nm: Range penalty parameter on near miss ranges.
        :type gamma_nm: float
        :param gamma_col: Range penalty parameter on collision ranges.
        :type gamma_col: float
        :param epsilon_course: Detectable course change. [deg/s]
        :type epsilon_course: float
        :param epsilon_speed: Detectable speed change. [m/s^2]
        :type epsilon_speed: float
        :param delta_chi_apparent: Course alteration considered readily apparent to other vessels. [deg]
        :type delta_chi_apparent: object
        :param delta_speed_apparent: Speed alteration considered readily apparent to other vessels [m/s]
        :type delta_speed_apparent: object
        :param delta_chi_md: Minimum detectable course change [deg]
        :type delta_chi_md: float
        :param delta_psi_md: Minimum detectable heading change [deg]
        :type delta_psi_md: float
        :param delta_speed_md: Minimum detectable speed change [m/s^2]
        :type delta_speed_md: float
        :param delta_speed_reduction_apparent: Apparent speed reduction threshold, value in [0,1]
        :type delta_speed_reduction_apparent: float
        :param alpha_critical_13: Angle defining an overtaking situation, when a vessel is approaching another from
        abaft the beam cf. rule 13.
        :type alpha_critical_13: float
        :param alpha_critical_14: Angle defining a head-on situation, when two vessels are approaching on reciprocal or
        nearly reciprocal courses cf. rule 14.
        :type alpha_critical_14: float
        :param alpha_critical_15: Angle defining a crossing situation cf. rule 15.
        :type alpha_critical_15: float
        :param gamma_ahead_13: Penalty parameter defining the severity of an ahead passing in an overtaking situation
        :type gamma_ahead_13: float
        :param gamma_course_cpa_14: Penalty parameter defining the severity of a port passing in a head-on situation
        :type gamma_course_cpa_14: float
        :param gamma_ahead_15: Penalty parameter defining the severity of an ahead passing in a crossing situation
        :type gamma_ahead_15: float
        :param gamma_delay_14: Penalty parameter defining the severity of a delayed maneuver
        :type gamma_delay_14: float
        :param gamma_nsb: Penalty parameter defining the severity of a non-starboard maneuver
        :type gamma_nsb: float
        """
        # Coastline incorporation

        # try:
        #    self.coastline = pd.read_csv("coastline.csv", sep = ';')
        #    self.using_coastline = True
        # except:
        #    #print("Could not read coastline.csv")
        #    self.using_coastline = False 

        self.vessels = []
        # todo: Add checks for log file format
        if len(bag_path) != 0:
            pass
            # self.read_rosbag(bag_path)
        if len(AIS_path) != 0:
            self.read_AIS(AIS_path, ship_path)

        ## Just a slightly convoluted method of storing case name/code
        case_name = AIS_path.replace("-sec.csv", "")
        self.case_name = case_name
        for i in reversed(range(len(case_name))):
            if AIS_path[i] == "-":
                self.case_name = case_name.replace(case_name[i - len(case_name):], "")[-5:]
                break

        self.n_vessels = len(self.vessels)
        if self.n_vessels == 0:
            print("No vessels in file:", AIS_path)
            return

        # todo: find better way of getting equal number of messages
        self.n_msgs = self.vessels[0].n_msgs
        for vessel in self.vessels:
            self.n_msgs = min(self.n_msgs, vessel.n_msgs)

        # todo: Adapt distances to size of vessels. May need more of them if there are multiple vessels involved
        self.r_pref = r_pref
        self.r_min = r_min
        self.r_nm = r_nm
        self.r_col = r_col
        self.r_colregs = [r_colregs_2_max, r_colregs_3_max, r_colregs_4_max]
        # self.r_detect = 1.8 * self.r_pref  # Range to contact at time of detection
        self.r_detect = r_colregs_2_max  # Set detection time equal to time when COLREGS start applying
        # OBS! Changing this will effect initial conditions in several functions
        # todo Find a better way of defining r_detect, for sbmpc r_detect >>> r_pref for small vessels

        try:
            assert (gamma_nm + gamma_col <= 1), "gamma_nm + gamma_col > 1"
        except AssertionError as msg:
            print(msg)
            # todo: Add checks, eg. r_min > r_nm

        self.gamma_nm = gamma_nm
        self.gamma_col = gamma_col
        self.epsilon_course = np.deg2rad(epsilon_course)
        self.epsilon_speed = epsilon_speed
        self.delta_chi_app = np.deg2rad(delta_chi_apparent)
        self.delta_chi_md = np.deg2rad(delta_chi_md)
        self.delta_psi_md = np.deg2rad(delta_psi_md)
        self.delta_speed_md = delta_speed_md
        self.delta_speed_app = delta_speed_apparent
        self.delta_speed_red_app = delta_speed_reduction_apparent
        self.phi_OT_min = np.deg2rad(
            phi_OT_min)  # Minimum relative bearing for an encounter to be defined as an overtaking
        self.phi_OT_max = np.deg2rad(
            phi_OT_max)  # Maximum relative bearing for an encounter to be defined as an overtaking
        self.phi_SB_lim = np.deg2rad(phi_SB_lim)  # Defines a starboard turn in rule 14
        try:
            assert (s_r + s_theta == 1), "s_r + s_theta != 1"
        except AssertionError as msg:
            print(msg)
        self.s_r = s_r
        self.s_theta = s_theta
        # alpha variables are relative bearing of own ship as seen from obstacle
        self.alpha_crit_13 = np.deg2rad(alpha_critical_13)
        self.alpha_crit_14 = np.deg2rad(alpha_critical_14)
        self.alpha_crit_15 = np.deg2rad(alpha_critical_15)
        self.alpha_cpa_min_15 = np.deg2rad(alpha_cpa_min_15)
        self.alpha_cpa_max_15 = np.deg2rad(alpha_cpa_max_15)
        self.alpha_ahead_lim_13 = np.deg2rad(alpha_ahead_lim_13)
        self.alpha_cut = np.deg2rad(alpha_cut)
        self.beta_cut_min = np.deg2rad(beta_cut)
        self.beta_cut_max = np.deg2rad(360 - beta_cut)
        # gamma variables are penalty weights, representing the severity of a given action
        self.gamma_ahead_13 = gamma_ahead_13
        self.gamma_bearing_cpa_14 = gamma_course_cpa_14
        self.gamma_delay_14 = gamma_delay_14
        self.gamma_nsb_14 = gamma_nsb_14
        self.gamma_ahead_15 = gamma_ahead_15
        self.gamma_17_safety = gamma_17_safety
        self.gamma_17_port_turn = gamma_17_port_turn

        # Scores
        self.s_13 = np.ones([self.n_vessels, self.n_vessels])  # Score: overtaking situation
        self.s_14 = np.ones([self.n_vessels, self.n_vessels])  # Score: head on situation
        self.s_15 = np.ones([self.n_vessels, self.n_vessels])  # Score: crossing situation
        self.s_16 = np.ones([self.n_vessels, self.n_vessels])  # Score: give way vessel responsibility
        self.s_17 = np.ones([self.n_vessels, self.n_vessels])  # Score: stand on vessel responsibility
        self.s_safety = np.ones([self.n_vessels, self.n_vessels])  # Score: safety
        self.s_safety_r = np.ones([self.n_vessels, self.n_vessels])  # Score: safety wrt. range
        self.s_safety_theta = np.ones([self.n_vessels, self.n_vessels])  # Score: safety wrt pose
        self.s_14_ptp = np.ones([self.n_vessels, self.n_vessels])  # Score: port-to-port passing
        self.s_total_per_obst = np.ones([self.n_vessels, self.n_vessels])  # Total score wrt each obst

        # Penalties
        self.p_delay = np.zeros([self.n_vessels, self.n_vessels])  # Penalty: delayed action
        self.p_na_delta_chi = np.zeros([self.n_vessels, self.n_vessels])  # Check: non-apparent course changes
        self.p_na_delta_v = np.zeros([self.n_vessels, self.n_vessels])  # Check: non-apparent speed changes
        self.p_delta_chi = np.zeros([self.n_vessels, self.n_vessels])  # Penalty: course changes when stay-on vessel
        self.p_delta_v_up = np.ones([self.n_vessels, self.n_vessels])  # Penalty: speed increase when stay-on vessel
        self.p_delta_v_down = np.zeros(
            [self.n_vessels, self.n_vessels])  # Penalty: speed deincrease when stay-on vessel
        self.p_na_delta = np.zeros([self.n_vessels, self.n_vessels])  # Penalty: non-apparent maneuvers
        self.p_13_ahead = np.zeros([self.n_vessels, self.n_vessels])  # Penalty: passing ahead (binary value)
        self.p_14_nsb = np.zeros([self.n_vessels, self.n_vessels])  # Penalty: non starboard maneuver
        self.p_14_sts = np.zeros([self.n_vessels, self.n_vessels])  # Penalty: starboard-starboard passing
        self.p_15_ahead = np.zeros([self.n_vessels, self.n_vessels])  # Penalty: passing ahead (binary value)
        self.p_17_so_delta_chi = np.zeros([self.n_vessels, self.n_vessels])  # Penalty: stand on course change
        self.p_17_so_delta_v = np.zeros([self.n_vessels, self.n_vessels])  # Penalty: stand on speed change
        self.p_17_port_turn = np.zeros([self.n_vessels, self.n_vessels])  # Penalty: detected port turn

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

    def evaluate_ownship_behavior(self):
        self.evaluate_vessel_behavior(self.vessels[self.OWN_SHIP])

    def evaluate_vessel_behavior(self, vessel):
        """Based on Woerner's alg.4"""
        if not self.ranges_set:
            self.find_ranges()  # Theta_0, Theta_cpa, r_cpa
        if not self.relative_heading_set:
            self.find_relative_heading()  # alpha, beta
        # todo: Get the following functions to calculate values for all vessels
        self.find_course_and_speed_alteration(vessel)  # DeltaTheta, DeltaV
        self.score_safety(vessel)
        self.find_maneuver_detect_index(vessel)
        self.determine_situations(vessel)  # TODO: use! (only used in plotting atm)
        for obst in self.vessels:
            if obst.id != vessel.id:
                init_idx = self.detection_idx[vessel.id, obst.id]
                rule = self.entry_criteria(vessel, obst, init_idx)
                if rule == self.OP:  # Obstacle passed
                    continue
                    # Not relevant wrt. Woerner's version
                elif rule == self.OTGW:  # Overtaking situation - own ship is give way vessel
                    # R13/16
                    self.score_rule_16(vessel)
                    self.score_rule_13(vessel)
                elif rule == self.CRGW:  # Crossing situation - own ship is give way vessel
                    # R15/16
                    self.score_rule_16(vessel)
                    self.score_rule_15(vessel)
                elif rule == self.NAR:  # No applicable rules
                    # S_r
                    continue
                elif rule == self.CRSO:  # Crossing situation - own ship is stand on vessel
                    # R15/17
                    self.penalty_course_change(vessel)
                    self.penalty_speed_change(vessel)
                    self.score_rule_17(vessel)
                    self.score_rule_15(vessel)
                elif rule == self.OTSO:  # Overtaking situation - own ship is stand on vessel
                    self.penalty_course_change(vessel)
                    self.penalty_speed_change(vessel)
                    self.score_rule_17(vessel)
                    self.score_rule_13(vessel)
                elif rule == self.HO:  # Head on situation
                    # self.penalty_delay(vessel)
                    self.check_penalty_na_course_change(vessel)
                    self.score_rule_14(vessel)

        np.set_printoptions(precision=3)

    # Functions for calculating necessary parameters -------------------------------------------------------------------
    from AV_class_functions._parameter_calc import find_ranges, find_relative_heading, \
        find_maneuver_detect_index, find_course_and_speed_alteration

    # Functions for calculating penalties ------------------------------------------------------------------------------
    def penalty_delay(self, vessel):
        """
        Calculate penalty for delayed maneuver, i.e. failing to take action in ample time.
        This is Minne's implementation with and added max() wrt. what is stated in  his master thesis
        """

        for obst in self.vessels:
            if obst.id > vessel.id:
                obst_detect_idx = self.detection_idx[vessel.id, obst.id]
                cpa_idx = self.cpa_idx[vessel.id, obst.id]
            elif obst.id < vessel.id:
                obst_detect_idx = self.detection_idx[obst.id, vessel.id]
                cpa_idx = self.cpa_idx[obst.id, vessel.id]
            else:
                self.p_delay[vessel.id, obst.id] = 0
                continue
            maneuver_detect_idx = vessel.maneuver_detect_idx[obst.id]
            r_detect = self.ranges[vessel.id, obst.id, obst_detect_idx]
            r_maneuver = self.ranges[vessel.id, obst.id, maneuver_detect_idx]
            r_cpa = self.ranges[vessel.id, obst.id, cpa_idx]
            self.p_delay[vessel.id, obst.id] = max(0,
                                                   min(1, (r_detect - r_maneuver) / (r_detect - r_cpa)))

    def penalty_na_maneuver(self, vessel):
        """ Calculate penalty for non-apparent maneuver, based on Woerner's alg 13."""
        # todo: save theshold as parameter
        threshold = 0.3
        self.check_penalty_na_course_change(vessel)
        self.check_penalty_na_speed_change(vessel)

        for obst in self.vessels:
            if obst.id != vessel.id:
                if (self.p_na_delta_chi[vessel.id, obst.id] < threshold) or \
                        (self.p_na_delta_v[vessel.id, obst.id] < threshold):
                    self.p_na_delta[vessel.id, obst.id] = 0
                elif self.p_na_delta_v[vessel.id, obst.id] < threshold:
                    self.p_na_delta[vessel.id, obst.id] = self.p_na_delta_chi[
                        vessel.id, obst.id]
                else:
                    self.p_na_delta[vessel.id, obst.id] = self.p_na_delta_chi[vessel.id, obst.id] * \
                                                          self.p_na_delta_v[vessel.id, obst.id]

    def check_penalty_na_course_change(self, vessel):
        """
        Penalty for non-apparent course changes, suggested by P. Kristian E. Minne as a stricter alternative to the
        penalty function suggested by K. Woerner (alg 14), used when ownship is give-way vessel.
        """

        R_max = 0.5
        for obst in self.vessels:
            if obst.id == vessel.id:
                self.p_na_delta_chi[vessel.id, obst.id] = 0
            else:
                delta_course = self.delta_course_max[vessel.id, obst.id]
                self.p_na_delta_chi[vessel.id, obst.id] = R_max * max(0, 1 - delta_course ** 2 /
                                                                      self.delta_chi_app ** 2)

    def check_penalty_na_speed_change(self, vessel):
        """
        Penalty for non-apparent speed changes, suggested by P. Kristian E. Minne as an alternative to the penalty
        function defined by K. Woerner (alg.15), used when ownship is give-way vessel.
        """
        R_max = 0.5
        for obst in self.vessels:
            if obst.id == vessel.id:
                continue
            obst_detect_idx = self.detection_idx[vessel.id, obst.id]
            u_0 = vessel.speed[obst_detect_idx]
            d_u_rel = (u_0 - abs(self.delta_speed_max_red[vessel.id, obst.id])) / u_0
            if d_u_rel > self.delta_speed_red_app:
                self.p_na_delta_v[vessel.id, obst.id] = R_max * (self.delta_speed_red_app - d_u_rel) / \
                                                        self.delta_speed_red_app
            else:
                self.p_na_delta_v[vessel.id, obst.id] = 0

    def penalty_course_change(self, vessel):
        """Penalize course change, based on woerner's Alg.16, used when own_ship is stand-on vessel """
        # todo: set max value as param
        R_max = 0.5
        for obst in self.vessels:
            if obst.id == vessel.id:
                continue
            p = 0
            # Todo: Get this into the implementation
            # Check each detetcted maneuver between obstactle detection and CPA
            # for i, value in enumerate(vessel.maneuver_detect_idx):
            #     if value > self.cpa_idx[vessel.id, obst.id]:
            #         continue
            #     if abs(vessel.delta_course[i]) < self.delta_psi_md:
            #         continue
            #     elif abs(vessel.delta_course[i]) > self.delta_chi_app:
            #         p = R_max
            #     else:
            #         p = R_max*(abs(vessel.delta_course[i]) - self.delta_psi_md) / (self.delta_chi_app - self.delta_psi_md)

            if vessel.maneuver_detect_idx.size == 0:
                continue
            # For now: Only test if a maneuver is detected between obstacle detection and CPA
            elif vessel.maneuver_detect_idx[0] > self.cpa_idx[vessel.id, obst.id]:
                continue
            elif abs(self.delta_course_max[vessel.id, obst.id]) < self.delta_psi_md:
                continue
            elif abs(self.delta_course_max[vessel.id, obst.id]) > self.delta_chi_app:
                p = R_max
            else:
                p = R_max * (abs(self.delta_course_max[vessel.id, obst.id]) - self.delta_psi_md) / (self.delta_chi_app
                                                                                                    - self.delta_psi_md)
            self.p_delta_chi[vessel.id, obst.id] = p

    def penalty_speed_change(self, vessel):
        """Penalize speed change, based on Woerner's Alg.17, used when own_ship is stand-on vessel"""

        if vessel.maneuver_detect_idx.size == 0:
            return
        R_max = 0.5
        for obst in self.vessels:
            if obst.id == vessel.id:
                continue
            # For now: Only test if a maneuver is detected between obstacle detection and CPA
            elif vessel.maneuver_detect_idx[0] > self.cpa_idx[vessel.id, obst.id]:
                continue
            dv_fast = self.delta_speed_max[vessel.id, obst.id]
            dv_slow = abs(self.delta_speed_max_red[vessel.id, obst.id])
            dv_max = max(dv_fast, dv_slow)
            # todo: obstacle detection index can only be used while colregs range is equal to detection range
            obst_detection_idx = self.detection_idx[vessel.id, obst.id]
            v_max = vessel.speed[obst_detection_idx] + dv_fast
            v_0 = vessel.speed[obst_detection_idx]

            if dv_max > self.delta_speed_md:
                self.p_delta_v_up[vessel.id, obst.id] = (v_0 / v_max) ** 2  # Penalize speeding up
                self.p_delta_v_down[vessel.id, obst.id] = R_max * (dv_slow / v_0)  # Penalize slowing down

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
                if self.ranges[vessel.id, obst.id, i] > self.r_colregs[0]:
                    # If outside COLREGS range
                    if abs(self.situation_matrix[vessel.id, obst.id, i - 1]) == self.OTSO:
                        # Overtaking situation passed when vessels are out of range
                        self.situation_matrix[vessel.id, obst.id, i] = self.OP
                    else:
                        # No applicable rules
                        self.situation_matrix[vessel.id, obst.id, i] = self.NAR
                elif self.ranges[vessel.id, obst.id, i] <= self.r_colregs[0]:
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

        if (beta > self.phi_OT_min) and (beta < self.phi_OT_max) and (abs(alpha) < self.alpha_crit_13) \
                and (vessel.speed[i] < obst.speed[i]):
            # Own-ship is being overtaken by obstacle j and is the stand on vessel.
            self.situation_matrix[vessel.id, obst.id, i] = self.OTSO
        elif (alpha_360 > self.phi_OT_min) and (alpha_360 < self.phi_OT_max) \
                and (abs(beta_180) < self.alpha_crit_13) and (vessel.speed[i] > obst.speed[i]):
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
        #    coast_inside = self.coastline.loc[(self.coastline['lon'] > (max(vessel.stateLonLat[0,i], obst.stateLonLat[0,i]) - x_width)) &\
        #                                      (self.coastline['lon'] < (min(vessel.stateLonLat[0,i], obst.stateLonLat[0,i]) + x_width)) &\
        #                                      (self.coastline['lat'] > (max(vessel.stateLonLat[1,i], obst.stateLonLat[1,i]) - y_width)) &\
        #                                      (self.coastline['lat'] < (min(vessel.stateLonLat[1,i], obst.stateLonLat[1,i]) + y_width))].reset_index(drop = True)

        #   if len(coast_inside.index) != 0:
        #       vector_1 = np.array([vessel.stateLonLat[0,i] - obst.stateLonLat[0,i], vessel.stateLonLat[1,i] - obst.stateLonLat[1,i]])
        #       for lon, lat in zip(coast_inside.lon.tolist(), coast_inside.lat.tolist()):
        #           vector_2 = np.array([vessel.stateLonLat[0,i] - lon, vessel.stateLonLat[1,i] - lat])
        #           unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        #           unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        #           dot_product = np.dot(unit_vector_1, unit_vector_2)
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

    def filterOutNonCompleteSituations(self, vessel, obst):

        debug = False
        # if "STAVANGER" in vessel.name:
        #    debug = True

        if debug: print(vessel.name, " - ", obst.name)

        if self.cpa_idx[vessel.id, obst.id] == 0 or \
                self.cpa_idx[vessel.id, obst.id] >= np.where(np.isnan(vessel.speed) == False)[-1][-1]:
            self.situation_matrix[vessel.id, obst.id, :] = self.NAR
            return self.situation_matrix[vessel.id, obst.id, :]

        ## Filter Head on situations
        if self.HO in self.situation_matrix[vessel.id, obst.id, :]:
            start = np.argmax(self.situation_matrix[vessel.id, obst.id, :] == self.HO)  # start of sit
            obst_yaw_start = obst.state[2, start]
            end = start + np.argmax( \
                np.array([abs_ang_diff(a, obst_yaw_start) for a in obst.state[2, start:]]) > 0.5 * np.pi)
            end = end if end != start else self.n_msgs

            alpha_abs = abs(self.alpha[vessel.id, obst.id, start:end + 1])
            alpha_max = max(alpha_abs)
            if alpha_max < 0.75 * np.pi:
                area = self.situation_matrix[vessel.id, obst.id] == self.HO
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR
            else:
                beta_max = max(abs(self.beta[vessel.id, obst.id, start:end + 1]))

                if beta_max < 0.75 * np.pi:
                    area = self.situation_matrix[vessel.id, obst.id, :] == self.HO
                    self.situation_matrix[vessel.id, obst.id, area] = self.NAR

        ## Filter Overtake stay on situations
        if self.OTSO in self.situation_matrix[vessel.id, obst.id, :]:
            # Start index of situation
            start = np.argmax(self.situation_matrix[vessel.id, obst.id, :] == self.OTSO)

            # The end index of situation is set to either when the obstacle ship have turned 
            # 90 degrees from the initial yaw or the end of case if the difference is large enough
            obst_yaw_start = obst.state[2, start]
            end = start + np.argmax( \
                np.array([abs_ang_diff(a, obst_yaw_start) for a in obst.state[2, start:]]) > 0.5 * np.pi)
            end = end if end != start else self.n_msgs

            if debug: print("Situation (OT) start stop:", start, "--", end)

            alpha_abs = abs(self.alpha[vessel.id, obst.id, start:end + 1])
            alpha_min = min(alpha_abs)

            if alpha_min > 0.25 * np.pi or alpha_min == np.pi \
                    or self.cpa_idx[vessel.id, obst.id] < start \
                    or self.cpa_idx[vessel.id, obst.id] > end:
                area = self.situation_matrix[vessel.id, obst.id, :] == self.OTSO
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR
            else:
                beta_abs = abs(self.beta[vessel.id, obst.id, start:end + 1])
                beta_max = max(beta_abs)

                if beta_max < 0.75 * np.pi:
                    area = self.situation_matrix[vessel.id, obst.id, :] == self.OTSO
                    self.situation_matrix[vessel.id, obst.id, area] = self.NAR

            #### CPA Criteria ####

            alpha_cpa_abs = abs(self.alpha[vessel.id, obst.id, self.cpa_idx[vessel.id, obst.id]])
            beta_cpa_abs = abs(self.beta[vessel.id, obst.id, self.cpa_idx[vessel.id, obst.id]])

            if debug: print("Alpha: ", alpha_cpa_abs, ", beta: ", beta_cpa_abs)

            if alpha_cpa_abs < 0.25 * np.pi or alpha_cpa_abs > 0.75 * np.pi:
                area = self.situation_matrix[vessel.id, obst.id, :] == self.OTSO
                if debug:
                    print(area)
                    print(self.situation_matrix[vessel.id, obst.id, :])
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR
                if debug:
                    print(area)
                    print(self.situation_matrix[vessel.id, obst.id, :])

            if beta_cpa_abs < 0.25 * np.pi or beta_cpa_abs > 0.75 * np.pi:
                area = self.situation_matrix[vessel.id, obst.id, :] == self.OTSO
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR

        ## Filter Overtake give way situations
        if self.OTGW in self.situation_matrix[vessel.id, obst.id, :]:
            start = np.argmax(self.situation_matrix[vessel.id, obst.id, :] == self.OTGW)  # start of sit
            obst_yaw_start = obst.state[2, start]

            end = start + np.argmax( \
                np.array([abs_ang_diff(a, obst_yaw_start) for a in obst.state[2, start:]]) > 0.5 * np.pi)
            end = end if end != start else self.n_msgs

            alpha_abs = abs(self.alpha[vessel.id, obst.id, start:end + 1])
            alpha_max = max(alpha_abs)

            if alpha_max < 0.75 * np.pi or alpha_max == np.pi \
                    or self.cpa_idx[vessel.id, obst.id] < start \
                    or self.cpa_idx[vessel.id, obst.id] > end:
                area = self.situation_matrix[vessel.id, obst.id, :] == self.OTGW
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR
            else:
                beta_abs = abs(self.beta[vessel.id, obst.id, start:end + 1])
                beta_min = min(beta_abs)

                if beta_min > 0.25 * np.pi:
                    area = self.situation_matrix[vessel.id, obst.id, :] == self.OTGW
                    self.situation_matrix[vessel.id, obst.id, area] = self.NAR

            #### CPA Criteria ####
            alpha_cpa_abs = abs(self.alpha[vessel.id, obst.id, self.cpa_idx[vessel.id, obst.id]])
            beta_cpa_abs = abs(self.beta[vessel.id, obst.id, self.cpa_idx[vessel.id, obst.id]])

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

        # CROSSING #
        if (self.CRGW or self.CRSO) in self.situation_matrix[vessel.id, obst.id, :]:
            from scipy.spatial.distance import cdist

            XA = np.array(vessel.state[0:2, :]).transpose()
            XB = np.array(obst.state[0:2, :]).transpose()

            cdist = np.min(cdist(XA, XB, metric='euclidean'))

            if cdist > 100:
                area = (self.situation_matrix[vessel.id, obst.id, :] == self.CRGW) | (
                        self.situation_matrix[vessel.id, obst.id, :] == self.CRSO)
                self.situation_matrix[vessel.id, obst.id, area] = self.NAR

            if False:  # DEBUG
                print("cdist:", cdist)
                print(self.situation_matrix[vessel.id, obst.id, :])

        if debug:
            print(self.situation_matrix[vessel.id, obst.id, :])

        return self.situation_matrix[vessel.id, obst.id, :]

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

        def getVesselParams(vessel, obst):
            if not vessel.maneuvers_found:
                printer_on = True  # Toggle printer ####

                self.find_maneuver_detect_index(vessel)
                maneuver_flag = True
                multi_man = False

                if printer_on:
                    print("\n\n")
                    print("Maneuver start stop ", vessel.maneuver_start_stop)
                    print(" COLREGS start:", start_idx, " stop:", stop_idx)

                man_inside = ((start_idx < vessel.maneuver_start_stop[:, 0]) & (
                            stop_idx > vessel.maneuver_start_stop[:, 0])) if len(
                    vessel.maneuver_detect_idx) > 0 else np.array([], dtype=bool)
                man_inside = np.array(man_inside)

                if printer_on:
                    print(man_inside)
                    print("Maneuver_detect_idx: ", vessel.maneuver_detect_idx[man_inside])

                i = 0
                man_number = 0
                maneuver_idx = None

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

                        if printer_on:
                            print("R-start:", range_val_start)
                            print("R-stop:", range_val_stop)
                            print("R-diff:", range_diff)

                        if maneuver_idx is not None:
                            if range_diff < diff_man_dist:
                                continue

                        if printer_on:
                            print("NEW BEST")
                        maneuver_idx = vessel.maneuver_start_stop[i][0]
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
                            man_number = i
                            pre_man_dist, pre_man_t_cpa = calcPredictedCPA(vessel, obst,
                                                                           vessel.maneuver_start_stop[i][0])
                            post_man_dist, post_man_t_cpa = calcPredictedCPA(vessel, obst,
                                                                             vessel.maneuver_start_stop[i][1])
                            if np.isnan(pre_man_dist) or \
                                    np.isnan(post_man_dist):
                                break
                            diff_man_dist = post_man_dist - pre_man_dist
                            break
                        i += 1
                else:
                    maneuver_flag = False

                if pre_man_t_cpa is not None:
                    pre_man_t_cpa -= maneuver_idx

                if post_man_t_cpa is not None:
                    post_man_t_cpa -= maneuver_idx

                return maneuver_flag, man_number, maneuver_idx, multi_man, pre_man_dist, pre_man_t_cpa, post_man_dist, post_man_t_cpa

        maneuver_made_own, man_number_own, maneuver_idx_own, multi_man_own, pre_man_dist_own, pre_man_t_cpa_own, post_man_dist_own, post_man_t_cpa_own = getVesselParams(
            own_vessel, obst_vessel)
        maneuver_made_obst, man_number_obst, maneuver_idx_obst, multi_man_obst, pre_man_dist_obst, pre_man_t_cpa_obst, post_man_dist_obst, post_man_t_cpa_obst = getVesselParams(
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
            multi_man_own,
            pre_man_dist_own,
            pre_man_t_cpa_own,
            post_man_dist_own,
            post_man_t_cpa_own,
            maneuver_made_obst,
            man_number_obst,
            maneuver_idx_obst,
            multi_man_obst,
            pre_man_dist_obst,
            pre_man_t_cpa_obst,
            post_man_dist_obst,
            post_man_t_cpa_obst)

        return params

    def getParameters(self, vessel, obst, start_idx, stop_idx, maneuver_made_own, man_number_own, maneuver_idx_own,
                      multi_man_own, pre_man_dist_own, pre_man_t_cpa_own, post_man_dist_own, post_man_t_cpa_own,
                      maneuver_made_obst, man_number_obst, maneuver_idx_obst, multi_man_obst, pre_man_dist_obst,
                      pre_man_t_cpa_obst, post_man_dist_obst, post_man_t_cpa_obst):
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

        # OBST param
        if maneuver_made_obst:
            idx = man_number_obst
            delta_course_obst = np.rad2deg(obst.delta_course[man_number_obst])
            delta_speed_obst = obst.delta_speed[man_number_obst]
            r_maneuver_obst = self.ranges[obst.id, vessel.id, maneuver_idx_obst]
        else:
            maneuver_idx_obst = (stop_idx - start_idx) // 2 + start_idx
            idx = maneuver_idx_obst
            delta_course_obst = 0
            delta_speed_obst = 0
            r_maneuver_obst = self.ranges[vessel.id, obst.id, idx]

        params = Parameters(len(self.vessels), \
                            vessel.mmsi, \
                            obst.mmsi, \
                            vessel.name, \
                            obst.name, \
                            vessel.callsign, \
                            obst.callsign, \
                            vessel.length, \
                            obst.length, \
                            vessel.width, \
                            obst.width, \
                            vessel.type, \
                            obst.type, \
                            vessel.nav_status[start_idx:stop_idx + 1].mean(), \
                            obst.nav_status[start_idx:stop_idx + 1].mean(), \
                            own_speed, \
                            obst_speed, \
                            multi_man_own, \
                            maneuver_made_own, \
                            maneuver_idx_own, \
                            r_maneuver_own, \
                            pre_man_dist_own, \
                            pre_man_t_cpa_own, \
                            post_man_dist_own, \
                            post_man_t_cpa_own, \
                            delta_speed_own, \
                            delta_course_own, \
                            multi_man_own, \
                            maneuver_made_obst, \
                            maneuver_idx_obst, \
                            r_maneuver_obst, \
                            pre_man_dist_obst, \
                            pre_man_t_cpa_obst, \
                            post_man_dist_obst, \
                            post_man_t_cpa_obst, \
                            delta_speed_obst, \
                            delta_course_obst, \
                            alpha_start, \
                            beta_start, \
                            r_cpa, \
                            alpha_cpa, \
                            beta_cpa, \
                            lon_maneuver, \
                            lat_maneuver, \
                            colreg_type, \
                            single_colreg_type, \
                            time, \
                            date_cpa, \
                            start_idx, \
                            stop_idx)
        return params

    def read_AIS(self, AIS_path, ship_path):
        '''
        :param AIS_path, ship_path: Paths to AIS-data/ship_info-data
        '''
        ais_df = pd.read_csv(AIS_path, sep=';', dtype={'mmsi': 'uint32', 'sog': 'float16', \
                                                       'cog': 'float16'}, \
                             parse_dates=['date_time_utc'], infer_datetime_format=True, na_values='')
        if len(ship_path) != 0:
            ship_df = pd.read_csv(ship_path, sep=';', \
                                  parse_dates=['date_min', 'date_max'], infer_datetime_format=True)
            # ship_df = pd.read_csv(ship_path, sep = ';', dtype = {'mmsi': 'uint32', \
            #            'length': 'int16', 'width': 'int16', 'type': 'float16'}, \
            #            parse_dates = ['date_min', 'date_max'], infer_datetime_format = True)
        else:
            ship_df = []

        origin_buffer = 0.01
        boats = getListOfMmsiDf(ais_df)
        lon0 = min(ais_df.lon) - origin_buffer
        lat0 = min(ais_df.lat) - origin_buffer

        ## strip coastline of unnecessary points

        lon_max = max(ais_df.lon.tolist())
        lat_max = max(ais_df.lat.tolist())
        del ais_df

        # if self.using_coastline:
        #    self.coastline = self.coastline.loc[(self.coastline['lon'] > lon0) &\
        #                                    (self.coastline['lat'] > lat0) &\
        #                                    (self.coastline['lon'] < lon_max) &\
        #                                    (self.coastline['lat'] < lat_max)].reset_index(drop = True)

        for boat in boats:
            ship_idx = None
            if len(ship_df) != 0:
                if boat.at[0, 'mmsi'] in ship_df.mmsi.tolist():
                    name_df = ship_df.loc[ship_df.mmsi == boat.at[0, 'mmsi']]
                    if len(name_df) == 1:
                        name = name_df.name.tolist()[0]
                        ship_idx = name_df.index.tolist()[0]
                    else:
                        name = boat.at[0, 'mmsi']
                        for date in name_df.index:
                            if name_df.at[date, 'date_min'] <= boat.at[0, 'date_time_utc'] <= name_df.at[
                                date, 'date_max']:
                                name = str(name_df.at[date, 'name'])
                                ship_idx = date
                                break
                else:
                    name = boat.at[0, 'mmsi']
            else:
                name = boat.at[0, 'mmsi']
            vessel = Vessel(name, len(boat))
            vessel.stateDateTime = boat.date_time_utc.to_list()
            vessel.mmsi = boat.at[0, 'mmsi']
            if isinstance(vessel.name, str):
                vessel.length = ship_df.at[ship_idx, 'length']
                vessel.width = ship_df.at[ship_idx, 'width']
                vessel.type = ship_df.at[ship_idx, 'type']
                vessel.imo = ship_df.at[ship_idx, 'imo']
                vessel.callsign = ship_df.at[ship_idx, 'callsign']
            else:
                vessel.name = str(vessel.name)

            for j in boat.index:
                vessel.state[0, j] = getDisToMeter(lon0, lat0, boat.lon[j], lat0)
                vessel.state[1, j] = getDisToMeter(lon0, lat0, lon0, boat.lat[j])
                vessel.state[2, j] = np.deg2rad(-normalize_180_deg(boat.cog[j] - 90))
                vessel.debug_state[2, j] = boat.cog[j]
                vessel.state[3, j] = knots_to_mps(boat.sog[j]) * np.cos(vessel.state[2, j])
                vessel.state[4, j] = knots_to_mps(boat.sog[j]) * np.sin(vessel.state[2, j])

                if np.isnan(vessel.nan_idx[0]) or (j == 0):
                    vessel.nan_idx[0] = j
                if not np.isnan(vessel.state[2, j]):
                    vessel.nan_idx[1] = j

            vessel.stateLonLat[0, :] = boat.lon.tolist()
            vessel.stateLonLat[1, :] = boat.lat.tolist()

            k_t_mps = lambda x: knots_to_mps(x)
            vessel.speed[:] = list(map(k_t_mps, boat.sog[:]))

            vessel.true_heading[:] = boat.true_heading.tolist()
            # vessel.message_nr[:] = boat.message_nr.tolist()
            # vessel.nav_status[:] = boat.nav_status.tolist()
            vessel.msgs_idx[:] = boat.index[:]

            vessel.dT = (boat.at[1, 'date_time_utc'] - boat.at[0, 'date_time_utc']).total_seconds() * 10 ** -9
            vessel.travel_dist = np.linalg.norm(
                [vessel.state[0, vessel.nan_idx[1]] - vessel.state[0, vessel.nan_idx[0]], \
                 vessel.state[1, vessel.nan_idx[1]] - vessel.state[1, vessel.nan_idx[0]]])
            if vessel.travel_dist > 1000:

                # Calculate derivative of speed
                speed = np.array(vessel.speed)

                from scipy.ndimage.filters import gaussian_filter
                target_area = np.isnan(speed) == False
                speed[target_area] = gaussian_filter(speed[target_area], sigma=1)

                target_area = [np.logical_and(np.logical_and(target_area[i] == True, target_area[i + 2] == True),
                                              target_area[i + 1] == True) for i in range(len(target_area) - 2)]
                target_area = np.append(False, target_area)
                target_area = np.append(target_area, False)
                if speed.size >= 3:
                    vessel.speed_der[:] = [0 for i in range(len(vessel.speed))]
                    speed = speed[np.isnan(speed) == False]
                    try:
                        vessel.speed_der[target_area] = [np.dot([speed[i], speed[i + 1], speed[i + 2]], [-0.5, 0, 0.5])
                                                         for i in range(len(speed) - 2)]
                    except:
                        pass

                        # Calculate derivatives of yaw
                a = np.array(vessel.state[2, :])
                d = np.append([0], a[1:] - a[:-1], 0)

                d[np.isnan(d)] = 0
                d[abs(d) < np.pi] = 0
                d[d < -np.pi] = -2 * np.pi
                d[d > np.pi] = 2 * np.pi  # d is now 2pi or -2pi at jumps from pi to -pi or opposite

                s = np.cumsum(d, axis=0)  # sum of all previuos changes

                target_area = np.isnan(a) == False

                a[target_area] = a[target_area] - s[
                    target_area]  # this is to not have sudden changes from pi to -pi or opposite count as maneuvers
                from scipy.ndimage.filters import gaussian_filter
                a[target_area] = gaussian_filter(a[target_area], sigma=2)

                target_area = [np.logical_and(target_area[i] == True, True == target_area[i + 2]) for i in
                               range(len(target_area) - 2)]
                target_area = np.append(False, target_area)
                target_area = np.append(target_area, False)
                if a.size >= 3:
                    a = a[np.isnan(a) == False]
                    vessel.maneuver_der[0, :] = [0 for i in range(len(vessel.state[2, :]))]
                    vessel.maneuver_der[0, target_area] = [np.dot([a[i], a[i + 1], a[i + 2]], [-0.5, 0, 0.5]) for i in
                                                           range(len(a) - 2)]
                    vessel.maneuver_der[1, :] = [0 for i in range(len(vessel.state[2, :]))]
                    vessel.maneuver_der[1, target_area] = [np.dot([a[i], a[i + 1], a[i + 2]], [1, -2, 1]) for i in
                                                           range(len(a) - 2)]

                    target_area = [np.logical_and(target_area[i] == True, True == target_area[i + 2]) for i in
                                   range(len(target_area) - 2)]
                    target_area = np.append(False, target_area)
                    target_area = np.append(target_area, False)
                    vessel.maneuver_der[2, :] = [0 for i in range(len(vessel.state[2, :]))]
                    vessel.maneuver_der[2, target_area] = [
                        np.dot([a[i], a[i + 1], a[i + 2], a[i + 3], a[i + 4]], [-0.5, 1, 0, -1, 0.5]) for i in
                        range(len(a) - 4)]
                    vessel.maneuver_der[1, :] = [0 for i in range(len(vessel.state[2, :]))]
                    vessel.maneuver_der[1, target_area] = [
                        np.dot([a[i], a[i + 1], a[i + 2], a[i + 3], a[i + 4]], [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
                        for i in range(len(a) - 4)]

                self.vessels.append(vessel)
        for id_idx, vessel in enumerate(self.vessels):
            vessel.id = id_idx

        del ship_df

    # Plotting functions -----------------------------------------------------------------------------------------------

    def plot_trajectories2(self, show_trajectory_at=None, specific_obst_names=None, save=False, save_folder='img'):
        """Plots trajectories of all ships and the applicable rules wrt to own ship"""
        fig, ax1 = plt.subplots(nrows=1, ncols=1)

        anon = False
        debug = False

        # TRAJECTORY PLOT

        ax1.set_title('Trajectories')
        ax1.set_xlabel('East')
        ax1.set_ylabel('North')
        x_min = 9.0 * 10 ** 9 if not self.using_coastline else 90
        x_max = -9.0 * 10 ** 9 if not self.using_coastline else 0
        y_min = x_min if not self.using_coastline else 90
        y_max = x_max if not self.using_coastline else 0

        colors = cm.rainbow(np.linspace(0, 1, self.n_vessels))
        for vessel in self.vessels:
            if vessel.id != self.OWN_SHIP:
                if specific_obst_names is not None:
                    its_in = False
                    for name in specific_obst_names:
                        if str(name) in vessel.name:
                            its_in = True

                    if not its_in:
                        continue

            label = vessel.name

            x_min_temp = min(vessel.stateLonLat[0])
            x_max_temp = max(vessel.stateLonLat[0])
            x_min = min(x_min, x_min_temp)
            x_max = max(x_max, x_max_temp)
            y_min_temp = min(vessel.stateLonLat[1])
            y_max_temp = max(vessel.stateLonLat[1])
            y_min = min(y_min, y_min_temp)
            y_max = max(y_max, y_max_temp)

        from mpl_toolkits.basemap import Basemap
        margin = 0.05

        m = Basemap(projection='merc',
                    llcrnrlat=y_min - margin, urcrnrlat=y_max + margin,
                    llcrnrlon=x_min - margin, urcrnrlon=x_max + margin,
                    resolution='h', ax=ax1)

        for vessel in self.vessels:  # Plot trajectories for all vessels
            if vessel.id != self.OWN_SHIP:
                if specific_obst_names is not None:
                    its_in = False
                    for name in specific_obst_names:
                        if name in vessel.name:
                            its_in = True
                    if not its_in:
                        continue
            if vessel.id == self.OWN_SHIP:
                color = 'k'
                label = vessel.name if not anon else "Ownship"
                own_label = vessel.name if not anon else "Ownship"
                if show_trajectory_at is not None:
                    calcTra = calcTrajectory(vessel, show_trajectory_at)
                    x, y = m(calcTra[0, :], calcTra[1, :])
                    ax1.plot(x, y, '+', color=color)
            else:
                color = "tomato"
                label = vessel.name if not anon else "Obsacle ship"
                obst_label = vessel.name if not anon else "Obsacle ship"

            x, y = m(vessel.stateLonLat[0], vessel.stateLonLat[1])
            ax1.plot(x, y, '-+',
                     label=label,
                     color=color)

            x, y = m(vessel.stateLonLat[0, 0], vessel.stateLonLat[1, 0])
            ax1.scatter(x, y, marker='x', color=color)

            x, y = m(vessel.stateLonLat[0, list(vessel.maneuver_detect_idx)],
                     vessel.stateLonLat[1, list(vessel.maneuver_detect_idx)])

            if vessel.id == self.OWN_SHIP:
                p1 = ax1.scatter(x, y,
                                 marker='o', label='maneuver', color=color)
            else:
                p2 = ax1.scatter(x, y,
                                 marker='o', color=color)

            if vessel.id != self.OWN_SHIP:  # Mark CPA and plot situations
                c = plt.cm.Dark2(vessel.id * 15)
                cpa_idx = self.cpa_idx[self.vessels[self.OWN_SHIP].id, vessel.id]

                x_cpa, y_cpa = m(self.vessels[self.OWN_SHIP].stateLonLat[0, cpa_idx],
                                 self.vessels[self.OWN_SHIP].stateLonLat[1, cpa_idx])
                p3 = ax1.scatter(x_cpa, y_cpa, marker='D', color=c)

                ax1.scatter(x_cpa, y_cpa, marker='D', color='k')

                x_man, y_man = m(vessel.stateLonLat[0, list(vessel.maneuver_detect_idx)],
                                 vessel.stateLonLat[1, list(vessel.maneuver_detect_idx)])
                ax1.scatter(x_man, y_man, marker='o', color=color)

                x_cpa, y_cpa = m(vessel.stateLonLat[0, cpa_idx], vessel.stateLonLat[1, cpa_idx])
                p4 = ax1.scatter(x_cpa, y_cpa, marker='D', color=color)

        import matplotlib.lines as mlines
        from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple

        if 'own_label' not in locals():
            own_label = 'Ownship'
        if 'obst_label' not in locals():
            obst_label = 'Obstacle ship'

        own_line = mlines.Line2D([], [], color='k', marker='+',
                                 markersize=6, label=own_label)
        obs_line = mlines.Line2D([], [], color='tomato', marker='+',
                                 markersize=6, label=obst_label)

        try:
            ax1.legend([own_line, obs_line, (p1, p2), (p3, p4)], [own_label, obst_label, 'Maneuver', 'CPA'],
                       scatterpoints=1,
                       numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})
        except:
            if debug:
                print("ERROR:")
                print(specific_obst_names)
                print()
                for name in specific_obst_names:
                    for vessel in self.vessels:
                        print(vessel.name, str(name) in vessel.name)

                print("\t ", [v.name for v in self.vessels])
                print("\t ", [v.mmsi for v in self.vessels])

        m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
        m.drawmapboundary(fill_color="#DDEEFF")
        m.drawcoastlines()

        if save:
            save_name = "./" + save_folder + "/" + self.vessels[self.OWN_SHIP].name + "-" + specific_obst_names[
                0] + "-" + self.case_name + ".png"
            plt.savefig(save_name, bbox_inches="tight", pad_inches=0.1)
            return save_name
        else:
            plt.show()

        return ""

    def plot_case(self, own_name=None, obst_name=None):
        """Plots trajectories of all ships and the applicable rules wrt to own ship"""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))

        # TRAJECTORY PLOT
        # ax.set_title('Trajectories')
        ax.set_xlabel('East')
        ax.set_ylabel('North')

        colors = cm.rainbow(np.linspace(0, 1, self.n_vessels))
        x_min, y_min = 100, 100
        x_max, y_max = 0, 0

        for vessel, color in zip(self.vessels, colors):
            if own_name != None and obst_name != None:
                if vessel.name != own_name and vessel.name != obst_name:
                    continue

            x_min = min(x_min, min(vessel.stateLonLat[0]))
            x_max = max(x_max, max(vessel.stateLonLat[0]))
            y_min = min(y_min, min(vessel.stateLonLat[1]))
            y_max = max(y_max, max(vessel.stateLonLat[1]))

        from mpl_toolkits.basemap import Basemap
        m = Basemap(projection='merc',
                    llcrnrlat=y_min - 0.2, urcrnrlat=y_max,
                    llcrnrlon=x_min - 0.5, urcrnrlon=x_max + 0.5,
                    resolution='h', ax=ax)

        for vessel, color in zip(self.vessels, colors):
            if own_name != None and obst_name != None:
                if vessel.name != own_name and vessel.name != obst_name:
                    continue
            m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
            m.drawmapboundary(fill_color="#DDEEFF")
            m.drawcoastlines()

            x, y = m(vessel.stateLonLat[0], vessel.stateLonLat[1])
            ax.plot(x, y, '-+', label=vessel.mmsi, color=color)

            x_start, y_start = m(vessel.stateLonLat[0, 0], vessel.stateLonLat[1, 0])
            ax.scatter(x_start, y_start, marker='x', color=color)

        ax.set_title("Case - " + self.case_name, fontsize=10)
        # ax.legend()
        # plt.savefig("./caseim2/" + self.case_name + ".png")
        plt.show()

    def plot_trajectory_man(self):
        """Plots trajectories of all ships and the applicable rules wrt to own ship"""
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        ax1, ax2 = ax.flatten()

        # TRAJECTORY PLOT
        ax1.set_title('Trajectories')
        ax1.set_xlabel('East')
        ax1.set_ylabel('North')

        colors = cm.rainbow(np.linspace(0, 1, self.n_vessels))
        for vessel, color in zip(self.vessels, colors):
            if not self.using_coastline:
                break

            if vessel.id != self.OWN_SHIP:
                continue

            x_min = min(vessel.stateLonLat[0])
            x_max = max(vessel.stateLonLat[0])
            y_min = min(vessel.stateLonLat[1])
            y_max = max(vessel.stateLonLat[1])

            x_diff = x_max - x_min
            y_diff = y_max - y_min

            x_diff, y_diff = max(x_diff, y_diff), max(x_diff, y_diff)

            x_min -= x_diff / 10
            x_max += x_diff / 10

            y_min -= y_diff / 10
            y_max += y_diff / 10

            if vessel.id == self.OWN_SHIP:
                color = 'k'

            from mpl_toolkits.basemap import Basemap
            m = Basemap(projection='merc',
                        llcrnrlat=y_min, urcrnrlat=y_max,
                        llcrnrlon=x_min, urcrnrlon=x_max,
                        resolution='h', ax=ax1)

            m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
            m.drawmapboundary(fill_color="#DDEEFF")
            m.drawcoastlines()

            x, y = m(vessel.stateLonLat[0], vessel.stateLonLat[1])
            ax1.plot(x, y, '-+', label=vessel.name, color='k')

            x_start, y_start = m(vessel.stateLonLat[0, 0], vessel.stateLonLat[1, 0])
            ax1.scatter(x_start, y_start, marker='x', color=color)

            x_man, y_man = m(vessel.stateLonLat[0, list(vessel.maneuver_detect_idx)],
                             vessel.stateLonLat[1, list(vessel.maneuver_detect_idx)])
            ax1.scatter(x_man, y_man, marker='o', label='maneuver', color=color)

        ax1.legend(loc='best')

        #### ax 2 ####
        ax2.plot(self.vessels[self.OWN_SHIP].maneuver_der[0, :], label='First der')
        ax2.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                    self.vessels[self.OWN_SHIP].maneuver_der[0, self.vessels[self.OWN_SHIP].maneuver_detect_idx])
        ax2.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                    self.vessels[self.OWN_SHIP].maneuver_der[0, self.vessels[self.OWN_SHIP].maneuver_detect_idx], \
                    marker='X')

        ax2.plot(self.vessels[self.OWN_SHIP].maneuver_der[1, :], label='Second der')
        ax2.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                    self.vessels[self.OWN_SHIP].maneuver_der[1, self.vessels[self.OWN_SHIP].maneuver_detect_idx])
        ax2.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                    self.vessels[self.OWN_SHIP].maneuver_der[1, self.vessels[self.OWN_SHIP].maneuver_detect_idx], \
                    marker='X')

        ax2.plot(self.vessels[self.OWN_SHIP].maneuver_der[2, :], label='Third der')
        ax2.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                    self.vessels[self.OWN_SHIP].maneuver_der[2, self.vessels[self.OWN_SHIP].maneuver_detect_idx])
        ax2.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                    self.vessels[self.OWN_SHIP].maneuver_der[2, self.vessels[self.OWN_SHIP].maneuver_detect_idx], \
                    marker='X')

        ax2.grid()
        ax2.legend(loc='best')

        plt.suptitle("Case - " + self.case_name, fontsize=14)
        plt.show()

    def plot_trajec_col(self, specific_obst_names=None):
        """Plots trajectories of all ships and the applicable rules wrt to own ship"""
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        ax1, ax2 = ax.flatten()

        # TRAJECTORY PLOT
        ax1.set_title('Trajectories')
        ax1.set_xlabel('East')
        ax1.set_ylabel('North')

        x_min = 9.0 * 10 ** 9 if not self.using_coastline else 90
        x_max = -9.0 * 10 ** 9 if not self.using_coastline else 0
        y_min = x_min if not self.using_coastline else 90
        y_max = x_max if not self.using_coastline else 0

        colors = cm.rainbow(np.linspace(0, 1, self.n_vessels))
        for vessel, color in zip(self.vessels, colors):  # Plot trajectories for all vessels
            if not self.using_coastline:
                break

            if vessel.id != self.OWN_SHIP:
                if specific_obst_names is not None:
                    its_in = False
                    for name in specific_obst_names:
                        if name in vessel.name:
                            its_in = True
                    if not its_in:
                        continue

            if vessel.id != self.OWN_SHIP and not any(
                    x != 0 for x in self.situation_matrix[self.OWN_SHIP, vessel.id, :]):

                color = 'silver'
                label = ''
            else:
                label = vessel.name

            x_min_temp = min(vessel.stateLonLat[0])
            x_max_temp = max(vessel.stateLonLat[0])
            x_min = min(x_min, x_min_temp)
            x_max = max(x_max, x_max_temp)
            y_min_temp = min(vessel.stateLonLat[1])
            y_max_temp = max(vessel.stateLonLat[1])
            y_min = min(y_min, y_min_temp)
            y_max = max(y_max, y_max_temp)
            if vessel.id == self.OWN_SHIP:
                color = 'k'

            ax1.plot(vessel.stateLonLat[0], vessel.stateLonLat[1], '-+', label=label, color=color)
            ax1.scatter(vessel.stateLonLat[0, 0], vessel.stateLonLat[1, 0], marker='x', color=color)

            if vessel.id == self.OWN_SHIP:
                ax1.scatter(vessel.stateLonLat[0, list(vessel.maneuver_detect_idx)],
                            vessel.stateLonLat[1, list(vessel.maneuver_detect_idx)],
                            marker='o', label='maneuver', color=color)
            else:
                ax1.scatter(vessel.stateLonLat[0, list(vessel.maneuver_detect_idx)],
                            vessel.stateLonLat[1, list(vessel.maneuver_detect_idx)],
                            marker='o', color=color)

            if vessel.id != self.OWN_SHIP:  # Mark CPA and plot situations
                c = plt.cm.Dark2(vessel.id * 15)
                cpa_idx = self.cpa_idx[self.vessels[self.OWN_SHIP].id, vessel.id]
                if color != 'silver':
                    ax1.scatter(self.vessels[self.OWN_SHIP].stateLonLat[0, cpa_idx],
                                self.vessels[self.OWN_SHIP].stateLonLat[1, cpa_idx], marker='D', color=c)
                ax1.scatter(vessel.stateLonLat[0, cpa_idx], vessel.stateLonLat[1, cpa_idx], marker='D', color=c)
                ax2.plot(self.situation_matrix[self.OWN_SHIP, vessel.id],
                         label=(str(self.vessels[self.OWN_SHIP].name) + ' wrt. ' + str(label)) if label != '' else '',
                         color=color)

                ax1.scatter(vessel.stateLonLat[0, list(vessel.maneuver_detect_idx)],
                            vessel.stateLonLat[1, list(vessel.maneuver_detect_idx)],
                            marker='o', color=color)

                ax1.scatter(vessel.stateLonLat[0, cpa_idx], vessel.stateLonLat[1, cpa_idx], marker='D', color=color)

        for vessel, color in zip(self.vessels, colors):  # Plot trajectories for all vessels
            ## not using coastine
            if vessel.id != self.OWN_SHIP:
                if specific_obst_names is not None:
                    its_in = False
                    for name in specific_obst_names:
                        if name in vessel.name:
                            its_in = True
                    if not its_in:
                        continue
            if self.using_coastline:
                break
            if vessel.id != self.OWN_SHIP and not any(
                    x != 0 for x in self.situation_matrix[self.OWN_SHIP, vessel.id, :]):
                color = 'silver'
                label = ''
            else:
                label = vessel.name

            x_min_temp = min(vessel.state[0])
            x_max_temp = max(vessel.state[0])
            x_min = min(x_min, x_min_temp)
            x_max = max(x_max, x_max_temp)
            y_min_temp = min(vessel.state[1])
            y_max_temp = max(vessel.state[1])
            y_min = min(y_min, y_min_temp)
            y_max = max(y_max, y_max_temp)

            if vessel.id == self.OWN_SHIP:
                color = 'k'

            ax1.plot(vessel.state[0], vessel.state[1], '-+', label=label, color=color)
            ax1.scatter(vessel.state[0, 0], vessel.state[1, 0], marker='x', color=color)
            if vessel.id == self.OWN_SHIP:
                ax1.scatter(vessel.state[0, vessel.maneuver_detect_idx], vessel.state[1, vessel.maneuver_detect_idx],
                            marker='o', label='maneuver', color=color)
            else:
                ax1.scatter(vessel.state[0, vessel.maneuver_detect_idx], vessel.state[1, vessel.maneuver_detect_idx],
                            marker='o', color=color)

            if vessel.id != self.OWN_SHIP:  # Mark CPA and plot situations
                c = plt.cm.Dark2(vessel.id * 15)
                cpa_idx = self.cpa_idx[self.vessels[self.OWN_SHIP].id, vessel.id]
                ax1.scatter(self.vessels[self.OWN_SHIP].state[0, cpa_idx],
                            self.vessels[self.OWN_SHIP].state[1, cpa_idx], marker='D', color=c, label="CPA")
                ax1.scatter(vessel.state[0, cpa_idx], vessel.state[1, cpa_idx], marker='D', color=c)
                ax2.plot(self.situation_matrix[self.OWN_SHIP, vessel.id],
                         label=(str(self.vessels[self.OWN_SHIP].name) + ' wrt. ' + str(label)) if label != '' else '',
                         color=color)

                ax1.scatter(vessel.state[0, vessel.maneuver_detect_idx], vessel.state[1, vessel.maneuver_detect_idx],
                            marker='o', color=color)

                ax1.scatter(vessel.state[0, cpa_idx], vessel.state[1, cpa_idx], marker='D', color=color)

        from mpl_toolkits.basemap import Basemap
        m = Basemap(projection='merc',
                    llcrnrlat=y_min, urcrnrlat=y_max,
                    llcrnrlon=x_min, urcrnrlon=x_max,
                    resolution='h', ax=ax1)

        m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
        m.drawmapboundary(fill_color="#DDEEFF")
        m.drawcoastlines()

        diff = max(x_max - x_min, y_max - y_min) / 2 + 200
        x_mid = x_min + (x_max - x_min) / 2
        y_mid = y_min + (y_max - y_min) / 2

        ax1.legend(loc='best')
        ax2.legend(loc='best')

        ax2.set_ylim(-3.2, 3.2)
        labels = ['', 'OP', 'OT-GW', 'CR-GW', '-', 'CR-SO', 'OT-SO', 'HO']
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('index')

        plt.suptitle("Case - " + self.case_name, fontsize=14)
        plt.savefig('output.png', dpi=100, bbox_inches='tight', pad_inches=0.1)

        plt.show()

    def plot_trajectories(self, show_trajectory_at=None, specific_obst_names=None):
        """Plots trajectories of all ships and the applicable rules wrt to own ship"""
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 7))
        ax1, ax2, ax3, ax4 = ax.flatten()

        # TRAJECTORY PLOT
        ax1.set_title('Trajectories')
        ax1.set_xlabel('East')
        ax1.set_ylabel('North')
        x_min = 9.0 * 10 ** 9 if not self.using_coastline else 90
        x_max = -9.0 * 10 ** 9 if not self.using_coastline else 0
        y_min = x_min if not self.using_coastline else 90
        y_max = x_max if not self.using_coastline else 0

        colors = cm.rainbow(np.linspace(0, 1, self.n_vessels))
        for vessel, color in zip(self.vessels, colors):  # Plot trajectories for all vessels
            if not self.using_coastline:
                break

            if vessel.id != self.OWN_SHIP:
                if specific_obst_names is not None:
                    its_in = False
                    for name in specific_obst_names:
                        if name in vessel.name:
                            its_in = True
                    if not its_in:
                        continue

            if vessel.id != self.OWN_SHIP and not any(
                    x != 0 for x in self.situation_matrix[self.OWN_SHIP, vessel.id, :]):

                color = 'silver'
                label = ''
            else:
                label = vessel.name

            x_min_temp = min(vessel.stateLonLat[0])
            x_max_temp = max(vessel.stateLonLat[0])
            x_min = min(x_min, x_min_temp)
            x_max = max(x_max, x_max_temp)
            y_min_temp = min(vessel.stateLonLat[1])
            y_max_temp = max(vessel.stateLonLat[1])
            y_min = min(y_min, y_min_temp)
            y_max = max(y_max, y_max_temp)
            if vessel.id == self.OWN_SHIP:
                color = 'k'
                if show_trajectory_at is not None:
                    calcTra = calcTrajectory(vessel, show_trajectory_at)
                    ax1.plot(calcTra[0, :], calcTra[1, :], '+', color=color)

            ax1.plot(vessel.stateLonLat[0], vessel.stateLonLat[1], '-+', label=label, color=color)

            ax1.scatter(vessel.stateLonLat[0, 0], vessel.stateLonLat[1, 0], marker='x', color=color)
            if vessel.id == self.OWN_SHIP:
                ax1.scatter(vessel.stateLonLat[0, list(vessel.maneuver_detect_idx)],
                            vessel.stateLonLat[1, list(vessel.maneuver_detect_idx)],
                            marker='o', label='maneuver', color=color)
            else:
                ax1.scatter(vessel.stateLonLat[0, list(vessel.maneuver_detect_idx)],
                            vessel.stateLonLat[1, list(vessel.maneuver_detect_idx)],
                            marker='o', color=color)

            if vessel.id != self.OWN_SHIP:  # Mark CPA and plot situations
                c = plt.cm.Dark2(vessel.id * 15)
                cpa_idx = self.cpa_idx[self.vessels[self.OWN_SHIP].id, vessel.id]
                if color != 'silver':
                    ax1.scatter(self.vessels[self.OWN_SHIP].stateLonLat[0, cpa_idx],
                                self.vessels[self.OWN_SHIP].stateLonLat[1, cpa_idx], marker='D', color=c)
                ax1.scatter(vessel.stateLonLat[0, cpa_idx], vessel.stateLonLat[1, cpa_idx], marker='D', color=c)
                ax2.plot(self.situation_matrix[self.OWN_SHIP, vessel.id],
                         label=(str(self.vessels[self.OWN_SHIP].name) + ' wrt. ' + str(label)) if label != '' else '',
                         color=color)

                ax1.scatter(vessel.stateLonLat[0, list(vessel.maneuver_detect_idx)],
                            vessel.stateLonLat[1, list(vessel.maneuver_detect_idx)],
                            marker='o', color=color)

                ax1.scatter(vessel.stateLonLat[0, cpa_idx], vessel.stateLonLat[1, cpa_idx], marker='D', color=color)

        for vessel, color in zip(self.vessels, colors):  # Plot trajectories for all vessels
            ## not using coastine
            if vessel.id != self.OWN_SHIP:
                if specific_obst_names is not None:
                    its_in = False
                    for name in specific_obst_names:
                        if name in vessel.name:
                            its_in = True
                    if not its_in:
                        continue
            if self.using_coastline:
                break
            if vessel.id != self.OWN_SHIP and not any(
                    x != 0 for x in self.situation_matrix[self.OWN_SHIP, vessel.id, :]):
                color = 'silver'
                label = ''
            else:
                label = vessel.name

            x_min_temp = min(vessel.state[0])
            x_max_temp = max(vessel.state[0])
            x_min = min(x_min, x_min_temp)
            x_max = max(x_max, x_max_temp)
            y_min_temp = min(vessel.state[1])
            y_max_temp = max(vessel.state[1])
            y_min = min(y_min, y_min_temp)
            y_max = max(y_max, y_max_temp)
            if vessel.id == self.OWN_SHIP:
                color = 'k'

            ax1.plot(vessel.state[0], vessel.state[1], '-+', label=label, color=color)
            ax1.scatter(vessel.state[0, 0], vessel.state[1, 0], marker='x', color=color)
            if vessel.id == self.OWN_SHIP:
                ax1.scatter(vessel.state[0, vessel.maneuver_detect_idx], vessel.state[1, vessel.maneuver_detect_idx],
                            marker='o', label='maneuver', color=color)
            else:
                ax1.scatter(vessel.state[0, vessel.maneuver_detect_idx], vessel.state[1, vessel.maneuver_detect_idx],
                            marker='o', color=color)

            if vessel.id != self.OWN_SHIP:  # Mark CPA and plot situations
                c = plt.cm.Dark2(vessel.id * 15)
                cpa_idx = self.cpa_idx[self.vessels[self.OWN_SHIP].id, vessel.id]
                ax1.scatter(self.vessels[self.OWN_SHIP].state[0, cpa_idx],
                            self.vessels[self.OWN_SHIP].state[1, cpa_idx], marker='D', color=c)
                ax1.scatter(vessel.state[0, cpa_idx], vessel.state[1, cpa_idx], marker='D', color=c)
                ax2.plot(self.situation_matrix[self.OWN_SHIP, vessel.id],
                         label=(str(self.vessels[self.OWN_SHIP].name) + ' wrt. ' + str(label)) if label != '' else '',
                         color=color)

                ax1.scatter(vessel.state[0, vessel.maneuver_detect_idx], vessel.state[1, vessel.maneuver_detect_idx],
                            marker='o', color=color)

                ax1.scatter(vessel.state[0, cpa_idx], vessel.state[1, cpa_idx], marker='D', color=color)

        from mpl_toolkits.basemap import Basemap
        m = Basemap(projection='merc',
                    llcrnrlat=y_min, urcrnrlat=y_max,
                    llcrnrlon=x_min, urcrnrlon=x_max,
                    resolution='h', ax=ax1)

        m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
        m.drawmapboundary(fill_color="#DDEEFF")
        m.drawcoastlines()

        diff = max(x_max - x_min, y_max - y_min) / 2 + 200
        x_mid = x_min + (x_max - x_min) / 2
        y_mid = y_min + (y_max - y_min) / 2
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax2.set_ylim(-3.2, 3.2)
        labels = ['', 'OP', 'OT-GW', 'CR-GW', '-', 'CR-SO', 'OT-SO', 'HO']
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('index')

        #### ax 3 ####
        plot_type = 2  # 1 for yaw, 2 for speeds
        if plot_type == 1:
            for vessel, color in zip(self.vessels, colors):
                if vessel.id != self.OWN_SHIP:
                    if specific_obst_names is not None:
                        its_in = False
                        for name in specific_obst_names:
                            if name in vessel.name:
                                its_in = True
                        if not its_in:
                            continue
                if vessel.id == self.OWN_SHIP:
                    color = 'k'
                else:
                    # continue       ## to disable multiple plots
                    if vessel.id != self.OWN_SHIP and not any(
                            x != 0 for x in self.situation_matrix[self.OWN_SHIP, vessel.id, :]):
                        continue

                a = np.array(vessel.state[2, :])
                d = np.append([0], a[1:] - a[:-1], 0)

                d[np.isnan(d)] = 0
                d[abs(d) < np.pi] = 0
                d[d < -np.pi] = -2 * np.pi
                d[d > np.pi] = 2 * np.pi  # d is now 2pi or -2pi at jumps from pi to -pi or opposite

                s = np.cumsum(d, axis=0)  # sum of all previuos changes

                target_area = np.isnan(a) == False

                a[target_area] = a[target_area] - s[
                    target_area]  # this is to not have sudden changes from pi to -pi or opposite count as maneuvers
                ax3.plot(a, '-', color=color)
                from scipy.ndimage.filters import gaussian_filter
                a[target_area] = gaussian_filter(a[target_area], sigma=2)

                ax3.plot(a, '--', color=color)
                if vessel.id == self.OWN_SHIP:
                    ax3.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                                a[self.vessels[self.OWN_SHIP].maneuver_detect_idx],
                                color=color)

                    ax3.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                                a[self.vessels[self.OWN_SHIP].maneuver_detect_idx],
                                marker='X', color=color)

        elif plot_type == 2:
            for vessel, color in zip(self.vessels, colors):
                if vessel.id != self.OWN_SHIP:
                    if specific_obst_names is not None:
                        its_in = False
                        for name in specific_obst_names:
                            if name in vessel.name:
                                its_in = True
                        if not its_in:
                            continue
                if vessel.id != self.OWN_SHIP and not any(
                        x != 0 for x in self.situation_matrix[self.OWN_SHIP, vessel.id, :]):
                    continue
                if vessel.id == self.OWN_SHIP:
                    color = 'k'
                ax3.plot(vessel.speed, color=color)
                from scipy.ndimage.filters import gaussian_filter
                s = vessel.speed
                target_area = np.isnan(s) == False
                s[target_area] = gaussian_filter(s[target_area], sigma=1)
                ax3.plot(s, '--', color=color)

        ax3.grid()

        #### ax 4 ####
        plot_type = 1  # 1 for derivatives, 2 for angle in case file and 3 for alpha's from ownship's perspective
        if plot_type == 1:
            ax4.plot(self.vessels[self.OWN_SHIP].maneuver_der[0, :], label='First der')
            ax4.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                        self.vessels[self.OWN_SHIP].maneuver_der[0, self.vessels[self.OWN_SHIP].maneuver_detect_idx])
            ax4.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                        self.vessels[self.OWN_SHIP].maneuver_der[0, self.vessels[self.OWN_SHIP].maneuver_detect_idx], \
                        marker='X')

            ax4.plot(self.vessels[self.OWN_SHIP].maneuver_der[1, :], label='Second der')
            ax4.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                        self.vessels[self.OWN_SHIP].maneuver_der[1, self.vessels[self.OWN_SHIP].maneuver_detect_idx])
            ax4.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                        self.vessels[self.OWN_SHIP].maneuver_der[1, self.vessels[self.OWN_SHIP].maneuver_detect_idx], \
                        marker='X')

            ax4.plot(self.vessels[self.OWN_SHIP].maneuver_der[2, :], label='Third der')
            ax4.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                        self.vessels[self.OWN_SHIP].maneuver_der[2, self.vessels[self.OWN_SHIP].maneuver_detect_idx])
            ax4.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                        self.vessels[self.OWN_SHIP].maneuver_der[2, self.vessels[self.OWN_SHIP].maneuver_detect_idx], \
                        marker='X')

            ax4.plot(self.vessels[self.OWN_SHIP].speed_der[:], label='Speed')
            ax4.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                        self.vessels[self.OWN_SHIP].speed_der[self.vessels[self.OWN_SHIP].maneuver_detect_idx])
            ax4.scatter(self.vessels[self.OWN_SHIP].maneuver_detect_idx,
                        self.vessels[self.OWN_SHIP].speed_der[self.vessels[self.OWN_SHIP].maneuver_detect_idx], \
                        marker='X')

        elif plot_type == 2:
            ax4.plot(self.vessels[self.OWN_SHIP].debug_state[2, :])
        elif plot_type == 3:
            for vessel, color in zip(self.vessels, colors):
                if vessel.id != self.OWN_SHIP:
                    if specific_obst_names is not None:
                        its_in = False
                        for name in specific_obst_names:
                            if name in vessel.name:
                                its_in = True
                        if not its_in:
                            continue
                if not any(x != 0 for x in self.situation_matrix[self.OWN_SHIP, vessel.id, :]):
                    continue  # ignore non-colregs
                ax4.plot(self.alpha[self.vessels[self.OWN_SHIP].id, vessel.id, :], '--', color=color,
                         label='Alpha ' + vessel.name)
                ax4.plot(self.beta[self.vessels[self.OWN_SHIP].id, vessel.id, :], '-', color=color,
                         label='Beta ' + vessel.name)

            ax4.hlines(self.phi_OT_min, 0, self.n_msgs, color='g', label='b_min')
            ax4.hlines(self.phi_OT_max, 0, self.n_msgs, color='y', label='b_max')
            ax4.hlines(self.alpha_crit_13, 0, self.n_msgs, color='b', label='a_13')
            ax4.hlines(-self.alpha_crit_13, 0, self.n_msgs, color='b', label='a_13')

        ax4.grid()
        ax4.legend(loc='best')

        if True:
            fulltext = 'Maneuvers:'
            for i, m in enumerate(self.vessels[self.OWN_SHIP].maneuver_detect_idx):
                try:
                    fulltext += '\n %i AT=%i, c:%.2f, s:%.1f' % (
                        i, m, self.vessels[self.OWN_SHIP].delta_course[i], self.vessels[self.OWN_SHIP].delta_speed[i])
                except:
                    print(self.vessels[self.OWN_SHIP].maneuver_detect_idx)
                    print(self.vessels[self.OWN_SHIP].delta_course)

            ax2.text(ax2.get_xbound()[1] + 1, 1, fulltext, fontsize=14)
            plt.subplots_adjust(right=0.85)

        plt.suptitle("Case - " + self.case_name, fontsize=14)
        plt.show()

    def plot_encounter_scores(self, vessel, obst):
        fig, ax = plt.subplots()
        init_idx = self.detection_idx[vessel.id, obst.id]
        situation = self.situation_matrix[vessel.id, obst.id, init_idx]
        if situation == self.OTSO:
            scores = [self.s_safety[vessel.id, obst.id], self.s_13[vessel.id, obst.id], self.s_17[vessel.id, obst.id],
                      1 - self.p_delta_v_up[vessel.id, obst.id], self.p_delta_v_down[vessel.id, obst.id],
                      self.p_delta_chi[vessel.id, obst.id]]
            labels = [r"$\mathcal{S}_{safety}$", r"$S_{13}$", r"$S_{17}$", r"$P_{\Delta v_{up}}$",
                      r"$P_{\Delta v_{down}}$", r"$P_{\Delta\chi}$"]
            colors = ['b', 'g', 'g', 'r', 'r', 'r']
            title = "Scores and penalties for overtaking encounter, own ship is stand-on"
        elif situation == self.OTGW:
            scores = [self.s_safety[vessel.id, obst.id], self.s_13[vessel.id, obst.id], self.s_16[vessel.id, obst.id],
                      self.p_delay[vessel.id, obst.id], self.p_na_delta[vessel.id, obst.id],
                      self.p_na_delta_chi[vessel.id, obst.id], self.p_na_delta_v[vessel.id, obst.id],
                      self.p_13_ahead[vessel.id, obst.id]]
            labels = [r"$\mathcal{S}_{safety}$", r"$S_{13}$", r"$S_{16}$", r"$P_{delay}$", r"$P_{na-man}$",
                      r"$P_{na-\Delta\chi}$", r"$P_{na-\Delta v}$", r"$P_{13}^{ahead}$"]
            colors = ['b', 'g', 'g', 'r', 'r', 'r', 'r', 'r', ]
            title = "Scores and penalties for overtaking encounter, own ship is give-way"
        elif situation == self.CRGW:
            scores = [self.s_safety[vessel.id, obst.id], self.s_15[vessel.id, obst.id], self.s_16[vessel.id, obst.id],
                      self.p_delay[vessel.id, obst.id], self.p_na_delta[vessel.id, obst.id],
                      self.p_na_delta_chi[vessel.id, obst.id], self.p_na_delta_v[vessel.id, obst.id]]
            labels = [r"$\mathcal{S}_{safety}$", r"$S_{15}$", r"$S_{16}$", r"$P_{delay}$", r"$P_{na-man}$",
                      r"$P_{na-\Delta\chi}$", r"$P_{na-\Delta v}$"]
            colors = ['b', 'g', 'g', 'r', 'r', 'r', 'r']
            title = "Scores and penalties for crossing encounter, own ship is give-way"
        elif situation == self.CRSO:
            scores = [self.s_safety[vessel.id, obst.id], self.s_15[vessel.id, obst.id], self.s_17[vessel.id, obst.id],
                      1 - self.p_delta_v_up[vessel.id, obst.id], self.p_delta_v_down[vessel.id, obst.id],
                      self.p_delta_chi[vessel.id, obst.id]]
            labels = [r"$\mathcal{S}_{safety}$", r"$S_{15}$", r"$S_{17}$", r"$P_{\Delta v_{up}}$",
                      r"$P_{\Delta v_{down}}$", r"$P_{\Delta\chi}$"]
            colors = ['b', 'g', 'g', 'r', 'r', 'r']
            title = "Scores and penalties for crossing encounter, own ship is stand-on"
        elif situation == self.HO:
            scores = [self.s_safety[vessel.id, obst.id], self.s_14[vessel.id, obst.id],
                      self.s_14_ptp[vessel.id, obst.id], self.p_delay[vessel.id, obst.id],
                      self.p_na_delta[vessel.id, obst.id], self.p_na_delta_chi[vessel.id, obst.id],
                      self.p_na_delta_v[vessel.id, obst.id], self.p_14_nsb[vessel.id, obst.id]]
            labels = [r"$\mathcal{S}_{safety}$", r"$S_{14}$", r"$S_{14}^{ptp}$", r"$P_{delay}$", r"$P_{na-man}$",
                      r"$P_{na-\Delta\chi}$", r"$P_{na-\Delta v}$", r"$P_{nsb}$"]
            colors = ['b', 'g', 'g', 'r', 'r', 'r', 'r']
            title = "Scores and penalties for head-on encounter"
        else:
            scores = [1, 1, 1, 1, 1, 1, 1, 1]
            labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            colors = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k']
            title = 'No scores or penalties'
        x = np.arange(len(labels))
        width = 0.8
        ax.bar(x + width / 2, scores, color=colors)
        ax.set_xticks(x + width)
        ax.set_ylim(0, 1)
        ax.set_xticklabels(labels)
        ax.set_title(vessel.name + ": " + title)
        plt.show()

    def plot_speed(self, vessel):
        fig, ax = plt.subplots()
        ax.plot(vessel.speed, label=vessel.name)
        ax.legend(loc='best')
        plt.axvline(x=self.detection_idx[vessel.id, 1], color='red')
        plt.show()

    def plot_heading(self, vessel):
        fig, ax = plt.subplots()
        ax.plot(np.rad2deg(vessel.state[2]), label=vessel.name)
        axes = plt.gca()
        axes.set_ylim([0, 359])
        ax.legend(loc='best')
        plt.title('Heading')
        plt.show()

    # Plotting of score/penalty metrics - ------------------------------------------------------------------------------
    def plot_s_safety(self):
        step_len_rad = 0.01
        alpha_cpa = np.arange(-np.pi, np.pi, step_len_rad, dtype=float)
        beta_cpa = np.arange(0, 2 * np.pi, step_len_rad, dtype=float)
        n_steps = len(alpha_cpa)
        r_min = 0
        r_max = self.r_colregs[0] + 100
        r_cpa = np.linspace(r_min, r_max, n_steps)

        s_alpha = np.zeros(n_steps)
        s_beta = np.zeros(n_steps)
        s_r = np.zeros(n_steps)

        for i in range(0, n_steps):
            # Safety wrt range
            if r_cpa[i] >= self.r_min:
                s_r[i] = 1
            elif self.r_nm <= r_cpa[i] < self.r_min:
                s_r[i] = 1 - self.gamma_nm * ((self.r_min - r_cpa[i]) / (self.r_min - self.r_nm))
            elif self.r_col <= r_cpa[i] < self.r_nm:
                s_r[i] = 1 - self.gamma_nm - self.gamma_col * (
                        (self.r_nm - r_cpa[i]) / (self.r_nm - self.r_col))
            else:
                s_r[i] = 0

            # Safety wrt pose
            if abs(alpha_cpa[i]) < self.alpha_cut:
                s_theta_a = 1 - np.cos(alpha_cpa[i])
            else:
                s_theta_a = 1 - np.cos(self.alpha_cut)
            if beta_cpa[i] < self.beta_cut_min or beta_cpa[i] > self.beta_cut_max:
                s_theta_b = 1 - np.cos(beta_cpa[i])
            else:
                s_theta_b = 1 - np.cos(self.beta_cut_min)
            s_alpha[i] = s_theta_a
            s_beta[i] = s_theta_b

        s_alpha_v, s_beta_v = np.meshgrid(s_alpha, s_beta)
        s_theta = np.multiply(s_alpha_v, s_beta_v)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.contour3D(s_alpha_v, s_beta_v, s_theta, 100)
        ax.set_xlabel(r"$S_\alpha$")
        ax.set_ylabel(r"$S_\beta$")
        ax.set_zlabel(r"$S_\Theta$")
        ax.set_title(r"$S_\Theta$")
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        ax.set_zticks([0, 1])
        plt.show()

        def format_func(value, tick_number):
            # find number of multiples of pi/4
            N = int(np.round(4 * value / np.pi))
            if N == 0:
                return "0"
            elif N == 1:
                return r"$\pi/4$"
            elif N == 2:
                return r"$\pi/2$"
            elif N == 3:
                return r"${0}\pi/4$".format(N)
            elif N % 4 > 0:
                return r"${0}\pi/4$".format(N)
            else:
                return r"${0}\pi$".format(N // 4)

        fig, ax = plt.subplots(ncols=1, nrows=2)
        ax1 = ax[0]
        ax2 = ax[1]
        ax1.plot(alpha_cpa, s_alpha, label='alpha')
        ax2.plot(beta_cpa, s_beta, label='beta')
        ax1.set_title(r"$S^\alpha_\Theta$")
        ax2.set_title(r"$S^\beta_\Theta$")
        ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
        ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax1.set_xlim(-np.pi, np.pi)
        ax2.set_xlim(0, 2 * np.pi)
        plt.show()

        fig, ax = plt.subplots(ncols=1, nrows=1)
        x_ticks = [self.r_col, self.r_nm, self.r_min, self.r_pref]
        x_labels = [r"$R_{col}$", r"$R_{nm}$", r"$R_{min}$", r"$R_{pref}$"]
        ax.plot(r_cpa, s_r)
        ax.set_title(r"$S_r$")
        ax.set_xlim(0, self.r_pref + 50)
        plt.xticks(x_ticks, x_labels)
        plt.show()

    def plot_p_delay(self):
        # r_detect = self.r_detect
        r_detect = 1.8 * self.r_pref
        r_cpa = self.r_min
        r_max = int(round(r_detect + 100))
        p_delay = np.zeros(r_max)
        p_delay_M = np.zeros(r_max)
        R_delay = np.zeros(r_max)
        for r_maneuver in range(r_max):
            p_delay[r_maneuver] = max(0, min(1, (r_detect - r_maneuver) / (r_detect - r_cpa)))  # Hagen's version
            p_delay_M[r_maneuver] = min(1, (r_detect - r_maneuver) / (r_detect - r_cpa))  # Minne's version
            R_delay[r_maneuver] = (r_detect - r_maneuver) / (r_detect - r_cpa)  # Woerner's version

        fig, ax = plt.subplots()
        ax.plot(p_delay)
        ax.set_title(r"$P_{delay}$")
        plt.axvline(x=r_cpa, color='red')
        plt.axvline(x=r_detect, color='green')
        ax.set_ylim(-0, 1.1)
        x_ticks = [r_cpa, r_detect]
        x_labels = [r"$r_{cpa}$", r"$r_{detect}$"]
        plt.xticks(x_ticks, x_labels)
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(R_delay)
        ax.set_title(r"$P_{delay}$")
        plt.axvline(x=r_cpa, color='red')
        plt.axvline(x=r_detect, color='green')
        ax.set_ylim(0, 1.1)
        x_ticks = [r_cpa, r_detect]
        x_labels = [r"$r_{cpa}$", r"$r_{detect}$"]
        plt.xticks(x_ticks, x_labels)
        plt.show()

    def plot_p_delta_chi_app(self):
        """Plot penalty for non-readily apparent course change as a functioon of the angle of the course change"""
        step_len_rad = 0.01
        R_max = 1
        delta_chi = np.arange(np.deg2rad(-180), np.deg2rad(180), step_len_rad, dtype=float)
        n_steps = len(delta_chi)
        p_delta_chi_app_W = np.zeros(n_steps)
        p_delta_chi_app_M = np.zeros(n_steps)
        p_delta_chi_app_H = np.zeros(n_steps)
        for i in range(0, n_steps):  # Woerner
            if abs(delta_chi[i]) < self.delta_chi_md:
                p_delta_chi_app_W[i] = 0
            else:
                p_delta_chi_app_W[i] = max(0, R_max * (self.delta_chi_app - abs(delta_chi[i])) /
                                           (self.delta_chi_app - self.delta_chi_md))
            p_delta_chi_app_M[i] = max(0, 1 - delta_chi[i] ** 2 / self.delta_chi_app ** 2)  # Minne
            p_delta_chi_app_H[i] = max(0, 1 - delta_chi[i] ** 2 / self.delta_chi_app ** 2)  # Hagen

        fig, ax = plt.subplots()
        ax.plot(delta_chi, p_delta_chi_app_W, label=r"$\mathfrak{R}^{\Delta \theta_{app}}$ - Woerner")
        ax.plot(delta_chi, p_delta_chi_app_M, label=r"$P \Delta \chi_{app}$ - Minne")
        ax.plot(delta_chi, p_delta_chi_app_H, linestyle='--', label=r"$P \Delta \chi_{app}$ Hagen")
        ax.set_title("Penalty for non-readily apparent course change")
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc='best')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)  # these are matplotlib.patch.Patch properties

        textstr = r"$\mathfrak{R}^{\Delta \theta_{app}} = %.2f$" % (R_max,)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        plt.show()

    def plot_p_delta_u_app(self):
        """ Worner uses this function in ALg. 15 to check for non apparent speed changes, while Minne uses it to
        penalize apparent speed changes. The only distinction between the two is that while Minne sets new speed equal
        to the maximum absolute speed change between time of detection and time of CPA and Woerner sets it equal to
        speed after slowing (no info on how this value is obtained)"""

        u_0 = np.array([5, 10, 15], dtype=float)
        n_speeds = len(u_0)
        step_change = 0.1
        new_speed = np.arange(0, 25, step_change)
        n_steps = len(new_speed)
        p_delta_speed_M = np.zeros((n_speeds, n_steps), dtype=float)
        du_rel = np.zeros((n_speeds, n_steps), dtype=float)
        du = np.zeros((n_speeds, n_steps), dtype=float)
        for i in range(0, n_speeds):
            du[i] = new_speed - u_0[i]
            du_rel[i] = np.divide(u_0[i] - abs(du[i]), u_0[i])

            for j in range(0, n_steps):
                if 1 - self.delta_speed_md > du_rel[i, j] > self.delta_speed_red_app:
                    p_delta_speed_M[i, j] = 1 * abs(self.delta_speed_red_app - du_rel[i, j]) / self.delta_speed_red_app
                else:
                    p_delta_speed_M[i, j] = 0

        fig, ax = plt.subplots(n_speeds, 1)
        for i in range(0, n_speeds):
            ax[i].plot(new_speed, p_delta_speed_M[i], label=r"$u_0 = %.2f$" % (u_0[i]))
            ax[i].set_ylim([0, 0.7])
            ax[i].set_ylabel('Penalty')
            ax[i].legend(loc='best')

            ax[i].axvline(x=u_0[i] + self.delta_speed_md * u_0[i], linestyle='--', color='green')
            ax[i].text(u_0[i] + self.delta_speed_md * u_0[i] + 0.01, 0.62, r"$\Delta u_{md}$", rotation=0)
            ax[i].axvline(x=u_0[i] - self.delta_speed_md * u_0[i], linestyle='--', color='green')
            ax[i].text(u_0[i] - self.delta_speed_md * u_0[i] + 0.01, 0.62, r"$\Delta u_{md}$", rotation=0)
            ax[i].axvline(x=u_0[i] + self.delta_speed_red_app * u_0[i], linestyle='--', color='red')
            ax[i].text(u_0[i] + self.delta_speed_red_app * u_0[i] + 0.01, 0.62, r"$\Delta u_{ra}$", rotation=0)
            ax[i].axvline(x=u_0[i] - self.delta_speed_red_app * u_0[i], linestyle='--', color='red')
            ax[i].text(u_0[i] - self.delta_speed_red_app * u_0[i] + 0.01, 0.62, r"$\Delta u_{ra}$", rotation=0)
        ax[0].set_title("Penalty check for non-readily apparent speed change")
        ax[0].set_xlabel('Speed change')

        plt.show()


# Helper metods --------------------------------------------------------------------------------------------------------
def rotate(vec, ang):
    """
    :param vec: 2D vector
    :param ang: angle in radians
    :return: input vector rotated by the input angle
    """
    r_mat = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    rot_vec = np.dot(r_mat, vec)
    return rot_vec


def normalize_2pi(angle):
    """
    :param angle: Angle in radians
    :type angle: float
    :return: Angle in radians normalized to [0, 2*pi)
    """

    if np.isnan(angle):
        return np.nan
    else:
        while angle >= 2 * np.pi:
            angle -= 2 * np.pi
        while angle < 0:
            angle += 2 * np.pi
        return angle


def normalize_pi(angle):
    """
    :param angle: Angle in radians
    :type angle: float
    :return: Angle in radians normalized to [-pi, pi)
    """

    if np.isnan(angle):
        return np.nan
    else:
        while angle >= np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def normalize_360_deg(angle):
    """
    :param angle: Angle in degrees
    :type angle: float
    :return: Angle in degrees normalized to [0, 360)
    """

    if np.isnan(angle):
        return np.nan
    else:
        while angle >= 360:
            angle -= 360
        while angle < 0:
            angle += 360
        return angle


def normalize_180_deg(angle):
    """
    :param angle: Angle in degrees
    :type angle: float
    :return: Angle in degrees normalized to [-180, 180)
    """
    if np.isnan(angle):
        return np.nan
    else:
        while angle >= 180:
            angle -= 2 * 180
        while angle < -180:
            angle += 2 * 180
        return angle


def abs_ang_diff(minuend, subtrahend):
    """
    Returns the smallest difference between two angles
    :param minuend: Angle in [0,2*pi]
    :param subtrahend: Angle in [0,2*pi]
    :return: Angle in [0,2*pi]
    """

    if np.isnan(minuend) or np.isnan(subtrahend):
        return 2 * np.pi
    return np.pi - abs(abs(minuend - subtrahend) - np.pi)


def signed_ang_diff(minuend, subtrahend):
    """
    Returns the signed difference between two angles
    :param minuend: Angle in [0,2*pi]
    :param subtrahend: Angle in [0,2*pi]
    :return: Angle in [-2*pi, 2*pi]
    """

    diff = minuend - subtrahend
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff


def format_func(value, tick_number):
    # find number of multiples of pi/4
    N = int(np.round(4 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/4$"
    elif N == 2:
        return r"$\pi/2$"
    elif N == 3:
        return r"${0}\pi/4$".format(N)
    elif N % 4 > 0:
        return r"${0}\pi/4$".format(N)
    else:
        return r"${0}\pi$".format(N // 4)


def getMmsiList(df):
    """
    Returns a list of all different mmsi contained in DataFrame
    :param df: DataFrame containing AIS_data
    :type df: pandas DataFrame
    :return: List of mmsi [mmsi_1, mmsi_2,..., mmsi_n]
    """
    temp_list = df.mmsi.unique().tolist()
    return temp_list


def getListOfMmsiDf(df):
    """
    Returns a list of DataFrames, where each DataFrame contains AIS_data for a ship
    :param df: DataFrame containing AIS_data
    :type df: pandas DataFrame
    :return: List of mmsi [DF_mmsi_1, DF_mmsi_2,..., DF_mmsi_n]
    """
    mmsiList = getMmsiList(df)
    mmsiDfList = [df[df.mmsi == mmsi].reset_index(drop=True) for mmsi in mmsiList]
    return mmsiDfList


def knots_to_mps(knots):
    """
    Transform velocity from knots to m/s
    :type knots: float
    :return: Velocity given in m/s (float)
    """
    if np.isnan(knots) or (knots >= 102.3):
        return np.nan
    mps = knots * 1.852 / 3.6
    return mps


def knots_to_kmph(knots):
    """
    Transform velocity from knots to km/h
    :type knots: float
    :return: Velocity given in km/h (float)
    """
    if np.isnan(knots) or (knots >= 102.3):
        return np.nan
    kmph = knots * 1.852
    return kmph


def getDisToMeter(lon1, lat1, lon2, lat2, **kwarg):
    """
    Find the distance between two lon/lat - coordinates given in meters
    :type lon1, lat1, lon2, lat2: float
    :return: Distance given in meters (float)
    """
    if lon1 == lon2:
        d_simple = abs(lat1 - lat2) * 111040
        return round(d_simple, 1)

    if lat1 == lat2:
        d_simple = abs(lon1 - lon2) * 6362.132 * 1000 * np.pi * 2 * np.cos(np.deg2rad(lat1)) / 360
        return round(d_simple, 1)

    if np.isnan(lon1) or np.isnan(lat1) or np.isnan(lon2) or np.isnan(lat2):
        return np.nan
    R = 6362.132
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
    d = R * c * 1000

    return round(d, 1)


def convertSecondsToTime(seconds):
    """
    Convert total seconds to datetime-stamp
    :type seconds: int
    :return: Total time in h:m:s (datetime)
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return datetime.time(hour=hour, minute=minutes, second=seconds)


def calcTrajectory(vessel, index):
    """
    Calculates a projected trajectory of an vessel from an index
    """

    speed = vessel.speed[index]
    lonStart, latStart = vessel.stateLonLat[:, index]

    posNow = np.array([vessel.state[0, index], vessel.state[0, index]])
    posPre = np.array([vessel.state[0, index - 1], vessel.state[0, index - 1]])
    newestVector = posNow - posPre

    course = np.rad2deg(np.arctan2(newestVector[1], newestVector[0]))
    if course < 0:
        course += 360

    length = vessel.n_msgs - index

    time_passed = 0

    trajectoryLonLat = np.empty([2, length])

    for i in range(length):
        time_passed += 1
        distance = speed * time_passed * 60  # TODO: 1 minute hardcoded

        trajectoryLonLat[0, i] = lonStart + np.sin(np.deg2rad(course)) * distance * 360 / (
                6362.132 * 1000 * np.pi * 2 * np.cos(np.deg2rad(latStart)))
        trajectoryLonLat[1, i] = latStart + np.cos(np.deg2rad(course)) * distance / 111040

    return trajectoryLonLat


def calcPredictedCPA(vessel1, vessel2, index):
    ##################
    printer_on = False
    ##################

    vessel1_trajectory = calcTrajectory(vessel1, index)
    vessel2_trajectory = calcTrajectory(vessel2, index)

    distance = getDisToMeter(vessel1_trajectory[0, 0], vessel1_trajectory[1, 0], vessel2_trajectory[0, 0],
                             vessel2_trajectory[1, 0])
    dist = distance

    i = 0
    s = 0
    time_at_cpa = 0

    if printer_on:
        print("Vessel1: ", vessel1.name)
        print("Vessel2: ", vessel2.name)
        print("Index:", index)

    while i < vessel1.n_msgs - index - 1:
        i += 1
        s += 1

        dist = getDisToMeter(vessel1_trajectory[0, i], vessel1_trajectory[1, i], vessel2_trajectory[0, i],
                             vessel2_trajectory[1, i])

        if np.isnan(dist):
            continue

        if dist > distance:
            break

        if printer_on:
            print("\t", "*", dist)

        distance = dist
        time_at_cpa = i + index

    if printer_on:
        print(distance)
        import csv
        with open('sums.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(str(s))

    return distance, time_at_cpa


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
        self.width = 5
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
    start_idx: int
    stop_idx: int
