from create_patches import create_patches
from fast_ista import fast_ista
import numpy as np 

# Path to where the data is stored 
path = '/home/ecbm6040/learnable-ISTA/data/train/'

# Reading the text file that contains names of all the images
fp = open('/home/ecbm6040/learnable-ISTA/data/iids_train.txt')
lines = fp.read().splitlines() # Create a list containing all lines
fp.close()

# Set the hyper-parameters
num_patches = 10000
alpha = 0.1

# Read the dictionary
Wd = np.load('/home/ecbm6040/learnable-ISTA/Wd.npy')

# Create the patches
X = create_patches(path, lines, num_patches)

print('Data Created')

for i in range(40):
	Z[:, i*50000 : (i+1)*50000] = fast_ista(X[:, i*50000 : (i+1)*50000], Wd, alpha)

print('Labels Created')

# Save the data
np.save('/home/ecbm6040/learnable-ISTA/X_train.npy', X)
np.save('/home/ecbm6040/learnable-ISTA/Z_train.npy', Z)