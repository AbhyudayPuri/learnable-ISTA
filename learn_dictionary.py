import numpy as np
from utils.create_patches import create_patches
from utils.coordinate_descent import coordinate_descent

def learn_dictionary(num_epochs, path, lines, num_patches, alpha, lr):

	# Setting the dimensions for the dictionary
	n = 100
	m = 400 

	# Randomly initilialize the dictionary 
	Wd = np.random.normal(0, 1/n, (n, m))
	# Normalizing the columns of the dictionary to unit magnitude
	Wd = Wd / np.linalg.norm(Wd, axis=0)

	# Creating a list to store the loss for each epoch
	total_loss = []

	# Stores the gradient of the loss wrt W
	Wd_grad = 0

	# Repeat the iterative procedure as many times as specified by the user
	for j in range(num_epochs):
		# Cumulatively stores the loss over the mini-batch
		loss = 0
		# Create the mini-batch
		X = create_patches(path, lines, num_patches)
		[p, q] = X.shape
		# print('Begin Descent')
		Z = coordinate_descent(X, Wd, alpha)
		# print('End')
		loss = 0.5 * np.linalg.norm(X - np.matmul(Wd, Z))**2
		Wd_grad = np.matmul(np.matmul(Wd, Z) - X, np.transpose(Z))

		# # Iterating over the mini-batch
		# for i in range(q):
		# 	print('i: {}'.format(i))
		# 	Xp = X[:,i].reshape(-1,1)
		# 	Z = coordinate_descent(Xp, Wd, alpha)
		# 	loss += 0.5 * np.linalg.norm(Xp - np.matmul(Wd, Z))**2
		# 	Wd_grad += np.matmul(np.matmul(Wd, Z) - Xp, np.transpose(Z))

		# Storing the loss
		total_loss.append(loss)

		# Updating the dictionary Wd
		Wd = Wd - lr * Wd_grad
		# Re-normalizing the columns to have unit norm
		Wd = Wd / np.linalg.norm(Wd, axis=0)

		print('epoch: {}'.format(j))
		print('loss: {}'.format(total_loss[j]))

	return Wd, total_loss