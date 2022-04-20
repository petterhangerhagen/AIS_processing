import numpy as np
from helper_methods import *

def score_safety(self, vessel):
    """
    Calculate the safety score of vessel with regards to all other vessels.
    """

    # todo: adjust ranges according to size and speed of vessels
    for obst in self.vessels:
        if obst.id == vessel.id:
            self.s_safety[vessel.id, obst.id] = 1
        if vessel.id < obst.id:
            cpa_idx = self.cpa_idx[vessel.id, obst.id]
            r_cpa = self.ranges[vessel.id, obst.id, cpa_idx]

            # Safety wrt range
            if r_cpa >= self.r_min:
                s_r = 1
            elif self.r_nm <= r_cpa < self.r_min:
                s_r = 1 - self.gamma_nm * ((self.r_min - r_cpa) / (self.r_min - self.r_nm))
            elif self.r_col <= r_cpa < self.r_nm:
                s_r = 1 - self.gamma_nm - self.gamma_col * (
                        (self.r_nm - r_cpa) / (self.r_nm - self.r_col))
            else:
                s_r = 0

            # Safety wrt pose
            alpha_cpa = self.alpha_cpa[vessel.id, obst.id]
            beta_cpa = self.beta_cpa[vessel.id, obst.id]
            if abs(alpha_cpa) < self.alpha_cut:
                s_theta_a = 1 - np.cos(alpha_cpa)
            else:
                s_theta_a = 1 - np.cos(self.alpha_cut)
            if beta_cpa < self.beta_cut_min or beta_cpa > self.beta_cut_max:
                s_theta_b = 1 - np.cos(beta_cpa)
            else:
                s_theta_b = 1 - np.cos(self.beta_cut_min)
            s_theta = s_theta_a * s_theta_b

            s = self.s_r * s_r + self.s_theta * s_theta
            self.s_safety[(vessel.id, obst.id), (obst.id, vessel.id)] = s

def score_rule_13(self, vessel):
    """Calculate the score for the vessel's behavior in an overtaking situation"""
    for obst in self.vessels:
        if obst.id != vessel.id:
            alpha_cpa = self.alpha_cpa[vessel.id, obst.id]
            if abs(alpha_cpa) < self.alpha_ahead_lim_13:
                p_13_ahead = 1
            else:
                p_13_ahead = 0
            self.p_13_ahead[vessel.id, obst.id] = p_13_ahead

            obst_detect_idx = self.detection_idx[vessel.id, obst.id]

            if self.situation_matrix[vessel.id, obst.id, obst_detect_idx] > 0:  # vessel is stand on
                self.s_13[vessel.id, obst.id] = self.s_17[vessel.id, obst.id]
            else:
                self.s_13[vessel.id, obst.id] = self.s_16[
                                                    vessel.id, obst.id] \
                                                - self.gamma_ahead_13 * p_13_ahead

def score_rule_14(self, vessel):
    """Calculate the score for the vessel's behavior in a head on situation. Baseed on Woernerr's alg 8"""
    for obst in self.vessels:
        if obst.id != vessel.id:
            obst_detect_idx = self.detection_idx[vessel.id, obst.id]
            i_cpa = self.cpa_idx[vessel.id, obst.id]
            alpha_cpa = self.alpha_cpa[vessel.id, obst.id]
            beta_cpa = self.beta_cpa[vessel.id, obst.id]
            s_14_alpha_cpa = ((np.sin(alpha_cpa) + 1) / 2) ** 2
            s_14_beta_cpa = ((np.sin(beta_cpa) + 1) / 2) ** 2
            s_14_bearing_cpa = s_14_alpha_cpa * s_14_beta_cpa
            p_14_bearing_cpa = 1 - s_14_bearing_cpa
            self.p_14_sts[vessel.id, obst.id] = p_14_bearing_cpa
            self.s_14_ptp[vessel.id, obst.id] = s_14_bearing_cpa

            delta_chi_plus = 0
            for i in range(obst_detect_idx, i_cpa):
                delta_chi_plus = min(delta_chi_plus,
                                     normalize_pi(vessel.state[2, i] - vessel.state[2, obst_detect_idx]))

            if delta_chi_plus < self.phi_SB_lim:  # If starboard maneuver
                self.p_14_nsb[vessel.id, obst.id] = 0
            else:
                self.p_14_nsb[vessel.id, obst.id] = (delta_chi_plus / self.phi_SB_lim) ** 4

            self.s_14[vessel.id, obst.id] = (1 - self.gamma_nsb_14 * self.p_14_nsb[vessel.id, obst.id]
                                             - self.gamma_delay_14 * self.p_delay[vessel.id, obst.id]
                                             - self.gamma_bearing_cpa_14 * p_14_bearing_cpa) * \
                                            (1 - self.p_na_delta_chi[vessel.id, obst.id])

def score_rule_15(self, vessel):
    """Calculate the score for the vessel's behavior in a crossing situation"""
    s_15 = []
    p_15_ahead_temp = []
    for obst in self.vessels:
        if obst.id == vessel.id:
            s_15.append(1)
            p_15_ahead_temp.append(0)
        else:
            obst_detect_idx = self.detection_idx[vessel.id, obst.id]
            alpha_cpa = self.alpha_cpa[vessel.id, obst.id]

            if self.alpha_cpa_min_15 < alpha_cpa < self.alpha_cpa_max_15:
                p_15_ahead = 1
                p_15_ahead_temp.append(1)
            else:
                p_15_ahead = 0
                p_15_ahead_temp.append(1)

            if self.situation_matrix[vessel.id, obst.id, obst_detect_idx] > 0:  # vessel is stand on
                s_15.append(self.s_17[vessel.id, obst.id])
            else:
                s_15.append(self.s_16[vessel.id, obst.id] - self.gamma_ahead_15 * p_15_ahead)
    self.s_15[vessel.id] = np.array(s_15)
    self.p_15_ahead[vessel.id] = np.array(p_15_ahead)

def score_rule_16(self, vessel):
    """Calculate the score for the vessel's behavior as the give way vessel"""
    self.penalty_na_maneuver(vessel)
    for obst in self.vessels:
        if obst.id > vessel.id:
            s_16 = self.s_safety[vessel.id, obst.id] \
                   * (1 - self.p_delay[vessel.id, obst.id]) * (
                           1 - self.p_na_delta[vessel.id, obst.id])
        elif obst.id < vessel.id:
            s_16 = self.s_safety[obst.id, vessel.id] \
                   * (1 - self.p_delay[vessel.id, obst.id]) * (
                           1 - self.p_na_delta[vessel.id, obst.id])
        else:
            s_16 = 1

        self.s_16[vessel.id, obst.id] = s_16

def score_rule_17(self, vessel):
    for obst in self.vessels:
        if obst.id == vessel.id:
            continue
        s_safety = self.s_safety[vessel.id, obst.id]
        p_delta_chi = self.p_delta_chi[vessel.id, obst.id]
        p_delta_v_up = self.p_delta_v_up[vessel.id, obst.id]
        p_delta_v_down = self.p_delta_v_down[vessel.id, obst.id]
        self.s_17[vessel.id, obst.id] = (s_safety - p_delta_chi) * p_delta_v_up - p_delta_v_down

def score_rule_17_minne(self, vessel):
    """Calculate the score for the vessel's behavior as the stand-on vessel. Minnes's version"""
    # todo: Add separate behaviour for stage 3 and 4
    s_17_temp = []
    for obst in self.vessels:
        if obst.id == vessel.id:
            s_17_temp.append(1)
            continue

        obst_detect_idx = self.detection_idx[vessel.id, obst.id]
        delta_course = self.delta_course_max[vessel.id, obst.id]
        p_17_delta_course = abs((abs(delta_course) - self.delta_chi_md) / \
                                (self.delta_chi_app - self.delta_chi_md))
        p_17_delta_course = min(1, p_17_delta_course)

        # Woerner's implementation
        delta_speed_fast = self.delta_speed_max[vessel.id, obst.id]
        delta_speed_slow = abs(self.delta_speed_max_red[vessel.id, obst.id])
        delta_speed_max = max(delta_speed_fast, delta_speed_slow)
        r_max = 0.5  # Maximum penlaty for slowing
        r_rule = 1  # Can't find the value for this param, so I'll leave it for now.
        if delta_speed_max < self.delta_speed_md:
            p_17_delta_speed = 0
        else:
            u_0 = vessel.speed[obst_detect_idx]
            p_17_delta_speed = r_rule * (u_0 / delta_speed_max) ** 2
            p_17_delta_speed = p_17_delta_speed - r_max * (delta_speed_slow / u_0)

        p_17_delta_speed = min(1, p_17_delta_speed)

        chi_0 = vessel.state[2, obst_detect_idx]
        p_0 = vessel.state[0:2, obst_detect_idx]
        p_1 = vessel.state[0:2, self.cpa_idx[vessel.id, obst.id]]
        p = np.array([[np.cos(chi_0), np.sin(chi_0)], [np.sin(chi_0), np.cos(chi_0)]]).dot(p_1 - p_0)

        # todo: Test this on a scenario where a vessel turns port
        if p[1] > (vessel.width * 2):
            p_port_turn = 1
        else:
            p_port_turn = 0
        self.p_17_port_turn[vessel.id, obst.id] = p_port_turn

        i_maneuver_detect = vessel.maneuver_detect_idx[obst.id]
        p_17_delta = 1 - (self.ranges[vessel.id, obst.id, i_maneuver_detect]
                          - self.r_colregs[1]) / (self.r_colregs[1] - self.r_colregs[2]) ** 2
        s_17 = 1 - self.gamma_17_safety * (1 - self.s_safety[vessel.id, obst.id])

        if self.situation_matrix[vessel.id, obst.id, obst_detect_idx] > 0:  # vessel is stand on
            s_17 = s_17 - p_17_delta * (p_17_delta_speed + p_17_delta_course)
            self.p_17_so_delta_chi[vessel.id, obst.id] = p_17_delta_course
            self.p_17_so_delta_v[vessel.id, obst.id] = p_17_delta_speed

        s_17 = s_17 - self.gamma_17_port_turn * p_port_turn
        s_17_temp.append(s_17)

    self.s_17[vessel.id] = np.array(s_17_temp)
