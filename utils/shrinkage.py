import numpy as np

# Shrinkage Function
def shrink(V, alpha):
	
	###########################################################################
	# V ==> The input vector                                                  #   
	# alpha ==> The value in between the vector is shrinked and thresholded   #
	#                                                                         #
	# The algorithm runs element-wise on the vector V                         #
	###########################################################################
	
	h = np.sign(V) * np.maximum(np.abs(V)-alpha, 0)
	return h