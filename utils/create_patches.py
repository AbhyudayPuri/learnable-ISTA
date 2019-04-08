from sklearn.feature_extraction import image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2 as cv

def create_patches(path, lines, num_patches):
	# path = the path of where the data is stored
	# lines = a list that contains all the file names
	# num_patches = the number of patches will be generated from each image

	# Total number of images
	num_images = len(lines)
	# Creating an array to store the patched data
	output = np.zeros([num_images * num_patches, 10, 10, 3], dtype=np.uint8)

	for i in range(num_images):
		im = plt.imread(path + lines[i] + '.jpg')
		output[i*num_patches : (i+1)*num_patches,:,:,:] = image.extract_patches_2d(im, (10, 10), max_patches=num_patches)

	return output

path = '../data/train/'
fp = open('../data/iids_train.txt')
lines = fp.read().splitlines() # Create a list containing all lines
fp.close()
num_patches = 10

output = create_patches(path, lines, num_patches)
print(output[1].shape)
plt.imshow(output[1])
plt.show()
