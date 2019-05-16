import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import torchvision
from lista.lista import lista

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
num_epochs = 200
batch_size = 2000

# Stores the loss through out the entire training
training_loss = []

X = np.load('/home/ecbm6040/learnable-ISTA/data_lista/X_train.npy')
Z = np.load('/home/ecbm6040/learnable-ISTA/data_lista/Z_train.npy')

print('Data has been loaded')
print('--------------------------------------------------------------')

# Divide data into train, val, and test
X_train = X[:, 0:1500000]
Z_train = Z[:, 0:1500000]
print('Train set created')

X_val = X[:,1500000:1750000]
Z_val = Z[:,1500000:1750000]
print('Val set created')

X_test = X[:, 1750000:2000000]
Z_test = Z[:, 1750000:2000000]
print('Test set created')
print('--------------------------------------------------------------')

num_iter = X_train.shape[1] / batch_size

print('Begin Training')

for epoch in range(num_epochs):

	# Stores the loss for an entire mini-batch
	running_loss = 0.0

	# Shuffling the data before each epoch
	permutation = np.random.permutation(X_train.shape[1])
	X_shuffle = X_train[:, permutation]
	Z_shuffle = Z_train[:, permutation]

	# Generate the mini-batch
	for i in range(num_iter):
		# Zero the parameter gradients
		optimizer.zero_grad()
		
		X_batch = torch.from_numpy(X_train[:,i*batch_size : (i+1)*batch_size]).type(torch.FloatTensor)
		Z_batch = torch.from_numpy(Z_train[:,i*batch_size : (i+1)*batch_size]).type(torch.FloatTensor)

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

		# Print Loss
		running_loss += loss.item()

		if i % 10 == 0 and epoch != 0:    # print every 20 mini-batches
			print('epoch: {}, iteration: {}, loss: {}'.format(epoch, i, running_loss / 10))
			training_loss.append(running_loss)

	print('Epoch {} Done'.format(epoch))

	# Saving the model
	torch.save(net, '/home/ecbm6040/learnable-ISTA/pretrained_models/Network_1.pth')

loss_file = open('/home/ecbm6040/learnable-ISTA/pretrained_models/loss.txt', '+w') # open a file in write mode
for item in training_loss:    # iterate over the list items
   loss_file.write(str(item) + '\n') # write to the file
loss_file.close()   # close the file 










