from sklearn.feature_extraction import image
import numpy as np

# This function creates a batch of data from the entire dataset
def create_patches(path, lines, num_patches):
	# path = the path of where the data is stored
	# lines = a list that contains all the file names
	# num_patches = the number of patches will be generated from each image

	# output = A numpy array of dim num_images*num_patches X 10 X 10 X 3

	# Total number of images in our dataset
	num_images = len(lines)

	# Creating an array to store the patched data
	output = np.zeros([num_images * num_patches, 10, 10, 3], dtype=np.uint8)

	# Reading the images and storing the randomly generated patches
	for i in range(num_images):
		im = plt.imread(path + lines[i] + '.jpg')
		output[i*num_patches : (i+1)*num_patches,:,:,:] = image.extract_patches_2d(im, (10, 10), max_patches=num_patches)

	return output
