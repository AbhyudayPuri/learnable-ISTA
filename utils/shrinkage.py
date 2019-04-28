import numpy as np

# Shrinkage Function
def shrink(V, theta):
	
	###########################################################################
	# V ==> The input vector                                                  #   
	# theta ==> The value in between the vector is shrinked and thresholded   #
	#                                                                         #
	# The algorithm runs element-wise on the vector V                         #
	###########################################################################
	
	h = np.sign(V) * np.maximum(np.abs(V)-theta, 0)
	return h