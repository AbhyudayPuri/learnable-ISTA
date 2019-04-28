import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
from utils.coordinate_descent import coordinate_descent
from utils.shrinkage import shrink
from learn_dictionary import learn_dictionary
import numpy as np
from utils.create_patches import create_patches
from utils.prox_l1 import prox_l1

# Path to where the data is stored 
path = 'data/train/'

# Reading the text file that contains names of all the images
fp = open('data/iids_train.txt')
lines = fp.read().splitlines() # Create a list containing all lines
fp.close()

# The number of patches that will be randomly generated from each image
num_patches = 10
alpha = 0.01
num_epochs = 1000
lr = 0.01
beta = 0.9

# # Generate a mini-batch
# mini_batch = create_patches(path, lines, num_patches)

# print(mini_batch.shape)

# plt.imshow(mini_batch[:,1].reshape(10,10), cmap = 'gray')

Wd, loss = learn_dictionary(num_epochs, path, lines, num_patches, alpha, lr, beta)

np.savetxt('Wd.txt', Wd)
np.save('Wd.npy', Wd)

loss_file = open('loss.txt', 'w') # open a file in write mode
for item in loss:    # iterate over the list items
   loss_file.write(str(item) + '\n') # write to the file
loss_file.close()   # close the file 
