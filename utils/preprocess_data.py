from utilts.create_patches import create_patches
from utils.fast_ista import fast_ista

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
Wd = np.load('Wd.npy')

# Create the patches
X = create_patches(path, lines, num_patches)

# Save the data patches
np.save('/home/ecbm6040/learnable-ISTA/X_train.npy', X)

# Create the sparse code
Z[0:50000] = fast_ista(X[0:50000], Wd, alpha)
Z[50000:100000] = fast_ista(X[50000:100000], Wd, alpha)
Z[100000:150000] = fast_ista(X[100000:150000], Wd, alpha)
Z[150000:200000] = fast_ista(X[150000:200000], Wd, alpha)

np.save('/home/ecbm6040/learnable-ISTA/Z_train.npy', Z)