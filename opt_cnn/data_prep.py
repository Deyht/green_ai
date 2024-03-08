
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm
import cv2
import os

class_count = 12500
image_size = 256

if(not os.path.isdir("data")):
	os.system("wget https://share.obspm.fr/s/jqGotokdYDXTbDS/download/data_asirra.tar.gz")
	os.system("tar -xzf data_asirra.tar.gz")

raw_data_array = []
for i in range(0,class_count):
	patch = np.asarray(Image.open("data/PetImages/Cat/%d.jpg"%(i)))
	raw_data_array.append(patch)
for i in range(0,class_count):
	patch = np.asarray(Image.open("data/PetImages/Dog/%d.jpg"%(i)))
	raw_data_array.append(patch)

transform_prep = A.Compose([
		A.LongestMaxSize(max_size=image_size, interpolation=1, p=1.0),
		A.PadIfNeeded(min_width=image_size, min_height=image_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
	])

processed_im_array = np.zeros((class_count*2,image_size, image_size, 3), dtype="uint8")

for i in tqdm(range(0,2*class_count)):
	transformed = transform_prep(image=raw_data_array[i])
	patch_aug = transformed['image']
	processed_im_array [i,:,:,:] = patch_aug[:,:,:]

processed_im_array.tofile("asirra_bin_256.dat")
