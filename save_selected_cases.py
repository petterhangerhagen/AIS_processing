""" This script will save the cases of one csv file """
import pandas as pd
import os
import sys
import shutil

save_folder = 'encs_selected'
sec_increment = '-60-sec' #is in the name of all interpolated files
save_selected = True

try:
    csv_file = sys.argv[1]
except:
    print("Include name of csv file!")
    exit()

try:
    if not csv_file.endswith(".csv"):
        csv_file = csv_file + ".csv"

    df = pd.read_csv(csv_file, sep = ';')
    if save_selected:
        if 'img_class' not in df:
            print("Img_class not found in the csv file")
        else:
            df = df[df['img_class'] == True].reset_index(drop = True)
    case_names = df.case.to_list()

except Exception as e:
    print("Error reading file:", csv_file)
    print(e)
    exit()

listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk('.'):
    listOfFiles += [os.path.join(dirpath, filename) for filename in filenames]

listOfFiles = [filename for filename in listOfFiles if ('info' not in filename and sec_increment in filename and save_folder not in filename)]

listOfFiles = [filename for filename in listOfFiles if any(case in filename for case in case_names)]

print(listOfFiles)

for filename in listOfFiles:
    #print(filename[1:])
    f = filename.split('/')[-1]
    #input(f)
    #shutil.copy(filename[1:], '/' + save_folder + '/' + f, follow_symlinks=False)
    shutil.copy(filename[2:], save_folder)

