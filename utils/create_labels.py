import numpy as np

Z1 = np.load('/home/ecbm6040/learnable-ISTA/Z1_train.npy')
Z2 = np.load('/home/ecbm6040/learnable-ISTA/Z_8_14.npy')
Z1 = np.load('/home/ecbm6040/learnable-ISTA/Z_14_20.npy')

Z = np.zeros((400,2000000))

Z[:,0:800000] = Z1
Z[:, 800000:1400000] = Z2
Z[:, 1400000:2000000] = Z3

np.save('/home/ecbm6040/learnable-ISTA/Z_train.npy', Z)