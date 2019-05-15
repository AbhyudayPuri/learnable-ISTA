import numpy as np 
from shrinkage import shrink

def fast_ista(X, A, alpha=0.01):

	# Hyper-Parameters
	max_iter = 5000

	# Getting the number of basis vectors in our dictionary
	m = A.shape[1]

	# Creating the vector to store the sparse code
	Z = np.random.normal(0, 1/m, (m, X.shape[1]))

	# Creating the variable to store the value of the previous iterate
	Z_prev = Z

	# FISTA
	for i in range(max_iter):
		Z_aux = Z + (i / (i + 3)) * (Z - Z_prev)
		Z_prev = Z 

		V = Z_aux - (1 / np.linalg.norm(A, 2) ** 2) * np.matmul(np.transpose(A), np.matmul(A, Z_aux) - X)
		theta = alpha / (np.linalg.norm(A, 2) ** 2)

		Z = shrink(V, theta)
		if i%100 == 0:
			print(i)

	return Z