import numpy as np
from utils.create_patches import create_patches
from utils.coordinate_descent import coordinate_descent
import time
from utils.prox_l1 import prox_l1

def learn_dictionary(num_epochs, path, lines, num_patches, alpha, lr, beta):

	# Setting the dimensions for the dictionary
	n = 100
	m = 400 

	# Setting the number of iterations for the code update and dictionary update
	num_iter_Z = 20
	num_iter_W = 5

	# Randomly initilialize the dictionary 
	Wd = np.random.normal(0, 1/n, (n, m))
	# Normalizing the columns of the dictionary to unit magnitude
	Wd = Wd / np.linalg.norm(Wd, axis=0)

	# Momentum for the dictionary
	Wd_mom = np.zeros((n, m))

	# Creating a list to store the loss for each epoch
	total_loss = []

	# Repeat the iterative procedure as many times as specified by the user
	for j in range(num_epochs):

		# Start time of one epoch
		start = time.time()

		# Cumulatively stores the loss over the mini-batch
		loss = 0
		# Create the mini-batch
		X = create_patches(path, lines, num_patches)
		# [p, q] = X.shape

		# Vector to store the sparse code output
		Z = np.random.normal(0, 1/m, (m, X.shape[1]))

		for i in range(10):
			for k in range(num_iter_Z):
				Z = prox_l1(X, Z, Wd, alpha)

			for l in range(num_iter_W):
				# Wd_grad = np.matmul(np.matmul(Wd, Z) - X, np.transpose(Z)) / X.shape[1]
				Wd_grad = np.matmul(np.matmul(Wd, Z) - X, np.transpose(Z))
				# Wd = Wd - (1 / np.linalg.norm(Z, 2)**2) * Wd_grad
				Wd_mom = (beta * Wd_mom) + ((1 - beta) * Wd_grad)
				# Wd = Wd - lr * Wd_grad
				Wd = Wd - lr * Wd_mom
				# Re-normalizing the columns to have unit norm
				Wd = Wd / np.linalg.norm(Wd, axis=0)
		
		# loss = (0.5 * np.linalg.norm(X - np.matmul(Wd, Z))**2) / X.shape[1]
		loss = (0.5 * np.linalg.norm(X - np.matmul(Wd, Z))**2)
		# Storing the loss
		total_loss.append(loss)

		if (j+1) % 10 == 0 :
			num_iter_Z += 5

		# Update learning rate to decay
		# if (j+1) % 20  == 0:
		# 	lr *= beta
		# 	print('lr: {}'.format(lr))
		# lr *= 1 / (1 + beta * j)

		# End time of one epoch

		#####################################################################
		#                                                                   #
		# PALM                                                              #
		#                                                                   #
		##################################################################### 



		end = time.time()

		print('Average sparsity: {}'.format((Z.size - np.count_nonzero(Z)) / Z.size))

		print('epoch: {}'.format(j))
		print('loss: {}'.format(total_loss[j]))
		print('time taken to execute: {} seconds'.format(end - start))

	return Wd, total_loss