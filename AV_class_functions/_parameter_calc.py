from .helper_methods import *


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
    if vessel.maneuvers_searched:
        return

    if vessel.travel_dist < 1000:
        vessel.maneuver_detect_idx = np.array([])
        vessel.delta_course = vessel.delta_course([])
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
            continue
            # TODO: Figure out what to do with this unreachable statement.
            second_der_zeroes = np.concatenate([second_der_zeroes, [i]])

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
        np.sign(vessel.maneuver_der[2, i]) != np.sign(vessel.maneuver_der[2, i + 1]) or vessel.maneuver_der[2, i] == 0
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
    mask = np.array([1]*len(maneuver_idx_list), dtype=bool)

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


def find_course_and_speed_alteration(self, vessel):
    """
    Find max course and speed alteration between the detection of an obstacle and CPA for each obstacle.
    NOTE: This function is not robust towards noise. Since the function records the maximum deviation
    between the time of detection and time of cpa any outliers within this range will directly affect the recorded
    value. This will also lead to penalties.
    """
    for obst in self.vessels:
        if obst.id == vessel.id:
            continue
        elif obst.id > vessel.id:
            detect_idx = self.detection_idx[vessel.id, obst.id]
            cpa_idx = self.cpa_idx[vessel.id, obst.id]
        else:
            detect_idx = self.detection_idx[obst.id, vessel.id]
            cpa_idx = self.cpa_idx[obst.id, vessel.id]
        delta_course = 0
        delta_speed = 0
        delta_speed_red = 0
        for i in range(detect_idx, cpa_idx + 1):
            delta_course = max(delta_course, abs(vessel.state[2, detect_idx] - vessel.state[2, i]))
            delta_speed = max(delta_speed, vessel.speed[i] - vessel.speed[detect_idx])
            delta_speed_red = min(delta_speed_red, vessel.speed[i] - vessel.speed[detect_idx])
        self.delta_course_max[vessel.id, obst.id] = delta_course
        self.delta_speed_max[vessel.id, obst.id] = delta_speed
        self.delta_speed_max_red[vessel.id, obst.id] = delta_speed_red
