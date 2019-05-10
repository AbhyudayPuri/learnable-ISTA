import torch 
import torch.nn as nn
import torchvision
import numpy as np 
import torch.nn.functional as F

# Defining the network architecture
class lista(nn.Module):
	def __init__(self):
		super(lista, self).__init__()

		# Defining the dimensions for the network parameters
		self.n = 100
		self.m = 400

		self.W = torch.randn(self.m, self.n)
		self.S = torch.randn(self.m, self.m)
		self.soft_thresh = nn.Softshrink(lambd=0.5)

	def forward(self, x):

		# Initializing variables
		B = torch.matmul(W, x)
		Z = self.soft_thresh(B)

		# First Pass 
		Z = self.soft_thresh(B + torch.matmul(S, Z))

		# Second Pass
		Z = self.soft_thresh(B + torch.matmul(S, Z))

		# Third Pass
		Z = self.soft_thresh(B + torch.matmul(S, Z))

		return Z
