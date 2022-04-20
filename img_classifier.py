import cv2  # Import the OpenCV library
import pandas as pd  # Import Pandas library
import sys  # Enables the passing of arguments
""" A tool to manually classify situations
Use auto_plot to generate images before running this script

Requirements:
- OpenCV (used in conjunction with conda and python 3.6.13)  """


overwrite = False 

try:
    csv_file = sys.argv[1]
except:
    csv_file = 'rstudio.csv'
    print("Assuming name of file: ", csv_file)


df = pd.read_csv(csv_file, sep=';')
print(df)

if 'img_class' not in df:
    df['img_class'] = ""


list_of_files = df.img_name.to_list()
img_class = df.img_class.to_list()

print(list_of_files)


IMAGE_NAME = list_of_files[0] 
 

print("Controls \nt: True \nf: False \nu: Undo previous classification \nq: quit and save \nesc:quit without saving")
for i, INPUT_IMAGE in enumerate(list_of_files):
    if pd.isnull(INPUT_IMAGE):
        continue
    print("Nr:", i, " ", INPUT_IMAGE, str(img_class[i]))
    if str(img_class[i]) != 'nan' and not overwrite:
        continue
    image = cv2.imread(INPUT_IMAGE, -1)

    quit_viewing = False
    img_done = False

    while True:
        # Show image 'Image mouse':
        cv2.imshow("Case", image)
     
        # Continue until 'q' is pressed:
        k = cv2.waitKey(0)

        if k == 27:  # esc
            exit()

        if k == ord('q'):
            quit_viewing = True
            img_done = True

        if k == ord('t'):
            img_class[i] = 'True'
            img_done = True

        if k == ord('f'):
            img_class[i] = 'False'
            img_done = True 

        if k == ord('u') and i > 0:
            img_class[i-1] = 'nan'
            print("Undid:", list_of_files[i-1], " restart to re-classify")

        if img_done:
            cv2.destroyAllWindows()
            break

    if quit_viewing:
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
df['img_class'] = img_class

print(df)
 
df.to_csv(csv_file, index=None, sep=';')
 
exit()
