import pandas as pd
import numpy as np
import os

data = pd.DataFrame()
for root, directory, files in os.walk('./para/'):
    for i, filename in enumerate(files):
        if filename.endswith('.csv'):
            if 'Para' not in filename:
                continue

            if i % 100 == 0:
                print("File: ", i, " / ", len(files))
            # TODO: Check filter conditionals
            try:
                df = pd.read_csv('para/' + filename, sep=';', dtype={
                                'n_ships': int,
                                'own_mmsi': int,
                                'obst_mmsi': int,
                                'own_name': str,
                                'obst_name': str,
                                'own_callsign': str,
                                'obst_callsign': str,
                                'own_length': float,
                                'obst_length': float,
                                'own_width': float,
                                'obst_width': float,
                                'own_type': int,
                                'obst_type': int,
                                'own_nav_status': float,
                                'obst_nav_status': float,
                                'own_speed': float,
                                'obst_speed': float,
                                'multi_man_own': bool,
                                'maneuver_made_own': bool,
                                'maneuver_index_own': int,
                                'r_maneuver_own': float,
                                'pre_man_dist_own': float,
                                'post_man_dist_own': float,
                                'delta_speed_own': float,
                                'delta_course_own': float,
                                'multi_man_obst': bool,
                                'maneuver_made_obst': bool,
                                'maneuver_index_obst': int,
                                'r_maneuver_obst': float,
                                'pre_man_dist_obst': float,
                                'post_man_dist_obst': float,
                                'delta_speed_obst': float,
                                'delta_course_obst': float,
                                'alpha_start': float,
                                'beta_start': float,
                                'r_cpa': float,
                                'alpha_cpa': float,
                                'beta_cpa': float,
                                'lon_maneuver': float,
                                'lat_maneuver': float,
                                'COLREG': float,
                                'single_COLREG_type': bool,
                                'time': str,
                                'start_idx': int,
                                'stop_idx': int
                                }, parse_dates=['date_cpa'], infer_datetime_format=True)

                df = df[df['maneuver_index_own'] != df['stop_idx']]
                df = df[df['maneuver_index_own'] > df['start_idx']].reset_index(drop=True)
            except:
                continue
            if len(df) == 0:
                continue
            not_filt = []
            for k in range(len(df)):
                not_filt.append(len(df.loc[(df['own_mmsi'] == df['own_mmsi'][k]) & (
                            (df['start_idx'] <= df['stop_idx'][k]) & (df['stop_idx'] >= df['start_idx'][k]))]))

            df['COLREGS_not_filt'] = not_filt
            df = df.loc[df['own_speed'] >= 1]
            df = df.loc[df['obst_speed'] >= 1].reset_index(drop=True)

            filt = []
            for k in range(len(df)):
                filt.append(len(df.loc[(df['own_mmsi'] == df['own_mmsi'][k]) & (
                            (df['start_idx'] >= df['stop_idx'][k]) | (df['stop_idx'] >= df['start_idx'][k]))]))

            df['COLREGS_filt'] = filt
            data = data.append(df)

print(data)
data = data.loc[data.apply(lambda x:
                           x['r_cpa'] > 4*np.min(np.array([x['own_width'], x['obst_width']]))
                           if np.min(np.array([x['own_width'], x['obst_width']])) != 0
                           else x['r_cpa'] > 4, axis=1)]

data.to_csv('superPara.csv', sep=';', index=False)
