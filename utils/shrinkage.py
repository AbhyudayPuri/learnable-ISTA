import numpy as np

# Shrinkage Function
def shrink(V, alpha):
  h = np.sign(V) * np.maximum(np.abs(V)-alpha, 0)
  return h