import numpy as np
from utils.shrinkage import shrink

# Implements the coordinate descent algorithm for the lasso optimization
def coordinate_descent(X, Wd, alpha):
	
	##################################################################                                           
	# Wd ==> The dictionary matrix                                   #
	# X ==> The matrix with vectors as its columns for which we want #
	#       to recover the sparse code                               #
	# alpha ==> Penalty on the L1 term in the opitmization           #                                                    
	################################################################## 
	
	# Initialize S = I - Wd^T * Wd
	[n, m] = Wd.shape
	S = np.eye(m) - np.matmul(np.transpose(Wd), Wd)
	# Initialize B = Wd^T * X
	B = np.matmul(np.transpose(Wd), X)
	# Vector to store the sparse code output
	Z = np.zeros_like(B)

	# Store the number of times our algorithm iterates for
	num_iters = 0

	# Iterating until the algorithm converges
	while(True):
			Z_bar = shrink(B, alpha)

			# Index of max element of each column
			k = np.argmax(np.abs(Z-Z_bar), axis=0)
			# Used to index the columns of the matrix
			index = np.arange(k.shape[0])

			Z_diff = Z_bar[k, index] - Z[k, index]
			B = B + (S[:, k] * Z_diff)
			Z[k, index] = Z_bar[k, index]
			num_iters += 1

			# Check if the algorithm is converging
			if np.mean(np.abs(Z - Z_bar)) < 1e-3:
				break
			# elif num_iters > 1000:
			# 	break

	# The optimal sparse code
	Z = shrink(B, alpha)
	return Z