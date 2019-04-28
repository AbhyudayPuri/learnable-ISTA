import numpy as np 
from utils.shrinkage import shrink

def prox_l1(X, Z, A, alpha):
	
	###############################################################
	# This is the proximal map for the l1 minimization problem:   #
	# 0.5*||X - AZ||^2 + alpha*|Z|                                #
	# where ||.|| is the l2 norm and |.| is the l1 norm           #
	#                                                             #
	# X --> each column of X stores a single image patch          #
	# A --> the dictionary                                        #
	# Z --> the sparse code                                       #
	#                                                             #
	# This function performs a single iteration of the proximal   #
	# gradient descent                                            #
	###############################################################
	
	V = Z - (1 / np.linalg.norm(A, 2) ** 2) * np.matmul(np.transpose(A), np.matmul(A, Z) - X)
	theta = alpha / (np.linalg.norm(A, 2) ** 2)

	Z = shrink(V, theta)
	return Z