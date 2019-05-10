from sklearn.feature_extraction import image
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread


# This function creates a batch of data from the entire dataset
def create_patches(path, lines, num_patches):
	
	###########################################################################
	# path = the path of where the data is stored                             #
	# lines = a list that contains all the file names                         #
	# num_patches = the number of patches will be generated from each image   #
	#                                                                         #
	# output = A numpy array of dim [10*10 X num_images*num_patches]          #
	###########################################################################

	# Total number of images in our dataset
	num_images = len(lines)

	# Creating an array to store the patched data
	# output = np.zeros([num_images * num_patches, 10, 10, 3], dtype=np.uint8)
	output = np.zeros([num_images * num_patches, 10, 10], dtype=np.float64)

	# Reading the images and storing the randomly generated patches
	for i in range(num_images):
		im = imread(path + lines[i] + '.jpg')
		im_gray = rgb2gray(im)
		# output[i*num_patches : (i+1)*num_patches,:,:,:] = image.extract_patches_2d(im, (10, 10), max_patches=num_patches)
		output[i*num_patches : (i+1)*num_patches,:,:] = image.extract_patches_2d(im_gray, (10, 10), max_patches=num_patches)
	
	# Re-shaping the patches into the appropriate dimensions
	output = output.reshape((-1, num_images * num_patches))

	# Shuffling the data set to randomize the order of patches
	np.random.shuffle(np.transpose(output))
	
	return output
