import DataCrop as dc
import cv2
import os

#choose the target data set
process_target = "train"
#process_target = "test"

# Set the data path
Train_data_dir = './Train/'
Test_data_dir = './Test/'
SR_data_dir = './SR-data/'

#set data path and data list
data_dir = SR_data_dir
directory = os.fsencode(data_dir)
data_list = os.listdir(directory)

# Set parameters
up_scale = 3
crop_num = 10       #number of crops from each SR image
crop_size = 32

# Set write directory
target_dir  = Train_data_dir


#get the file list in the target folder
for file in data_list:
    filename = os.fsdecode(file)
    filepath = SR_data_dir + filename
    img = cv2.imread(filepath)

    dc.modCrop(img, up_scale);
    img = cv2.resize(img, dsize=(0,0) , fx= 1/up_scale, fy= 1/up_scale, interpolation= cv2.INTER_CUBIC)           #downsampling
    img = cv2.resize(img, dsize=(0,0), fx= up_scale, fy= up_scale, interpolation=cv2.INTER_CUBIC)          #upsampling

    # randomly cropping images and write to the target dir
    for i in range(crop_num):
        res = dc.randomCrop(img, size= crop_size)
        cv2.imwrite(target_dir + os.path.splitext(os.path.basename(filename))[0] + "_upscale" + str(up_scale) + "_cp" + str(i+1) + os.path.splitext(os.path.basename(filename))[1], res)





