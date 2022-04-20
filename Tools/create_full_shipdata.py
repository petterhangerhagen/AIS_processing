""" Used to generate a singe shipdata file named full_shipdata.csv  """
""" Assumes file structure month/date_shipdata.csv                  """
import os
import time
import pandas as pd
import numpy as np
from os import path
import sys
import psutil

#################################################################################

debug = False 

directory_path = './'
print('STARTING: full_shipdata.py')

df_full = pd.DataFrame()

for root, dirs, files in os.walk(directory_path):
    proc = []

    number_of_files = len(files)

    for count, filename in enumerate(files):
        if filename.endswith("_shipdata.csv"): 
            if "full" in filename:
                continue

            df = pd.read_csv(path.join(root, filename), sep = ";")

            df['date_min'] = [filename.replace("_shipdata.csv","")]*len(df) 
            df['date_max'] = [filename.replace("_shipdata.csv","")]*len(df) 

            if len(df_full) == 0:
                df_full = df
            else:
                df_full = pd.concat([df_full, df])

df_full = df_full.sort_values(by=['mmsi']).reset_index(drop=True)

for numb, mmsi in enumerate(df_full.mmsi.unique()):
    df_mmsi = df_full[df_full.mmsi == mmsi].sort_values(by=['date_min'])
    df_pre = df_mmsi

    if len(df_mmsi) <= 2:
        continue

    i = 0
    delete = [False]*len(df_mmsi)
    idx = []

    if debug: print(df_mmsi) 
    while i <= len(df_mmsi):
        df_pair = df_mmsi.iloc[i:i+2]

        if len(df_pair) == 1:
            break


        df_pair = df_pair.drop(columns = ['date_min', 'date_max']).fillna(0)
        eq = df_pair.iloc[0] == df_pair.iloc[1]

        if all(eq):
            df_mmsi.date_max.iat[i] = df_mmsi.date_min.iat[i+1]
            index = df_mmsi.index.to_list()[i+1]
            idx.append(index)
            df_mmsi = df_mmsi.drop(index)
        else:
            i += 1
            if debug: print(df_pair)

    df_full = df_full.drop(idx)
    df_full.loc[df_mmsi.index.to_list()] = df_mmsi

    if debug: print("######")
    if debug: print(df_pre)
    if debug: print("POST")
    if len(df_mmsi) != 1:
        if debug: input(df_mmsi)
    else:
        if debug: print(df_mmsi)

    if numb % (len(df_full.mmsi.unique()) // 100) == 0:
        print("At: ", numb, "/", len(df_full.mmsi.unique()))
    

df_full.to_csv('full_shipdata.csv', sep=";")

print("Done creating full_shipdata.csv")

