from create_patches import create_patches
from fast_ista import fast_ista
import numpy as np 
import time

# Set the hyper-parameters
alpha = 0.1

# Read the input data
X = np.load('/home/ecbm6040/learnable-ISTA/X_train.npy')
X = X[:,1400000:2000000]
# Read the dictionary
Wd = np.load('/home/ecbm6040/learnable-ISTA/Wd.npy')


Z = np.zeros((400, X.shape[1]))
print(Z.shape)

print('Data Created')

for i in range(60):
	start = time.time()
	print('Data Chunk {}/80'.format(i+1))
	Z[:, i*10000 : (i+1)*10000] = fast_ista(X[:, i*10000 : (i+1)*10000], Wd, alpha)
	end = time.time()
	np.save('/home/ecbm6040/learnable-ISTA/Z2_train.npy', Z)
	print("Time taken for 1 mini-batch: {}".format(end-start))

print('Labels Created')

# Save the data
# np.save('/home/ecbm6040/learnable-ISTA/X_train.npy', X)
np.save('/home/ecbm6040/learnable-ISTA/Z2_train.npy', Z)