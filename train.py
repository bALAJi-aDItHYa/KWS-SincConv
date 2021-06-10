import torch
import torch.nn as nn
from SincNet_Model import SincNet
from load_data import data_ret
import numpy as np

def train():
	running_loss = 0
	N = len(train_loader)

	# Switch the model to train mode
	model.train()

	for i, sample_batched in enumerate(train_loader):
		optimizer.zero_grad()

		inp = torch.autograd.Variable(sample_batched['data'].cuda())
		label = torch.autograd.Variable(sample_batched['label'].cuda())
		print(inp.view(8,1,16000).shape)

		output = model(inp.view(8,1,16000))

		#Loss
		loss = criterion(output, label)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

	print("")





if __name__ == '__main__':
	epochs = 60
	batch_size=8

	model = SincNet().cuda()
	print('model created')

	#Training params
	optimizer = torch.optim.Adam(model.parameters(), 0.001)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
	
	train_loader, val_loader, test_loader = data_ret()
	criterion = nn.NLLLoss()

	for e in range(epochs):

		optimizer.step()
		scheduler.step()
		train()
		validate()