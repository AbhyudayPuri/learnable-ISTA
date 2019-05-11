import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import torchvision
from utils.create_patches import create_patches
from lista.lista import lista
from utils.fast_ista import fast_ista

# Creating the network
net = lista()

# Checking if there is a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device being used =', device)

# Transferring the network onto the GPU
net.to(device)

# Ensuring that the model is in the training mode
net.train()

# Choosing the loss function criteria
criterion = nn.MSELoss()    # This is the l2 Loss Function

# Choosing the optimizer and its hyper-parameters
optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)    # Adaptive Momentum Optimizer

# Hyper-Parameters
num_epochs = 1000
num_patches = 10
alpha = 0.01

# Stores the loss through out the entire training
training_loss = []

path = './data/train/'

# Reading the text file that contains names of all the images
fp = open('./data/iids_train.txt')
lines = fp.read().splitlines() # Create a list containing all lines
fp.close()

# Loading the dictionary 
Wd = np.load('./Wd.npy')

for epoch in range(num_epochs):

	print(epoch)

	# Zero the parameter gradients
	optimizer.zero_grad()

	# Stores the loss for an entire mini-batch
	running_loss = 0.0

	# Generate the mini-batch
	X = create_patches(path, lines, num_patches)
	print('Batch Generated')

	# Generate the ground truth for these patches
	Z = fast_ista(X, Wd, alpha)

	X = torch.from_numpy(X).type(torch.FloatTensor)
	Z = torch.from_numpy(Z).type(torch.FloatTensor)
	print('Ground Truth Generated')

	# Pushing onto GPU
	X = X.to(device)
	Z = Z.to(device)

	# Forward Pass
	prediction = net(X)

	# Computng the loss
	loss = criterion(prediction, Z)
	# loss = (prediction - Z).pow(2).sum()

	# Back Propogation    
	loss.backward()
	
	# Updating the network parameters
	optimizer.step()

	print('Epoch Done')

	# Print Loss
	running_loss += loss.item()

	if epoch % 10 == 0 and epoch != 0:    # print every 20 mini-batches
		print('epoch: {}, loss: {}'.format(epoch, running_loss))
		training_loss.append(running_loss)

	# Saving the model
	torch.save(net, './pretrained_models/Network_1.pth')

loss_file = open('./pretrained_models/loss.txt', 'w') # open a file in write mode
for item in training_loss:    # iterate over the list items
   loss_file.write(str(item) + '\n') # write to the file
loss_file.close()   # close the file 










