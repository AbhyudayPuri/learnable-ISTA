import numpy as np
from utils.shrinkage import shrink

# Implements the coordinate descent algorithm for the lasso optimization
def coordinate_descent(X, Wd, alpha):
  [n, m] = Wd.shape
  S = np.eye(m) - np.matmul(np.transpose(Wd), Wd)
  # Initialize 
  B = np.matmul(np.transpose(Wd), X)
  Z = np.zeros_like(B)
  [m, n] = B.shape

  # Iterating until the algorithm converges
  num_iters = 0
  while(True):
      Z_bar = shrink(B, alpha)
      k = np.argmax(np.abs(Z-Z_bar))
      B = B + np.multiply(S[:, k].reshape(10,1), (Z_bar[k] - Z[k]))
      Z[k] = Z_bar[k]
      num_iters += 1
      
      if np.sum(np.abs(Z - Z_bar)) < 1e-5:
          break
          
  Z = shrink(B, alpha)
  print('The algorithm converges in {} iterations'.format(num_iters))
  return Z