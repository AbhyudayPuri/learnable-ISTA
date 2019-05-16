import torch 
import torch.nn as nn
import torchvision
import numpy as np 
import torch.nn.functional as F
from torch.autograd import Variable


# Defining the network architecture
class lista(nn.Module):
	def __init__(self):
		super(lista, self).__init__()

		# Defining the dimensions for the network parameters
		self.n = 100
		self.m = 400

		# self.W = torch.nn.Parameter(torch.randn((self.m, self.n), requires_grad=True))
		self.W = torch.nn.Parameter(torch.from_numpy(np.random.normal(0, 1/self.m, (self.m, self.n))), requires_grad=True)
		self.S = torch.nn.Parameter(torch.from_numpy(np.random.normal(0, 1/self.m, (self.m, self.m))), requires_grad=True)
		# self.S = torch.nn.Parameter(torch.randn((self.m, self.m), requires_grad=True))
		self.soft_thresh = nn.Softshrink(lambd=0.01)



	def forward(self, x):

		# Initializing variables
		B = torch.matmul(self.W, x)
		Z = self.soft_thresh(B)

		# First Pass 
		Z = self.soft_thresh(B + torch.matmul(self.S, Z))

		# Second Pass
		Z = self.soft_thresh(B + torch.matmul(self.S, Z))

		# Third Pass
		Z = self.soft_thresh(B + torch.matmul(self.S, Z))

		return Z
