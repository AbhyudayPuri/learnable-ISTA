from create_patches import create_patches
import numpy as np 

# Set the hyper-parameters
num_patches = 10000

# Path to where the data is stored 
path = '/home/ecbm6040/learnable-ISTA/data/train/'

# Reading the text file that contains names of all the images
fp = open('/home/ecbm6040/learnable-ISTA/data/iids_train.txt')
lines = fp.read().splitlines() # Create a list containing all lines
fp.close()

# Create the patches
X = create_patches(path, lines, num_patches)
print(X.shape)
np.save('/home/ecbm6040/learnable-ISTA/X_train.npy', X)