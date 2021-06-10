# ----------------------------------------------------------------------------
# train.py
# Written by Balaji Adhithya V
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn
from SincNet_Model import SincNet
from load_data import data_ret
import numpy as np
import time

epochs = 60
batch_size=16

model = SincNet().cuda()
print('model created')

IN_PATH = "/home/balaji5199/Desktop/KWS/try1.pth"
model.load_state_dict(torch.load(IN_PATH))

#Training params
optimizer = torch.optim.Adam(model.parameters(), 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_loader, val_loader, test_loader = data_ret()
criterion = nn.NLLLoss()

for e in range(epochs):
	running_loss = 0
	N = len(train_loader)

	# Switch the model to train mode
	model.train()
	t0 = time.time()

	for i, sample_batched in enumerate(train_loader):
		optimizer.zero_grad()

		inp = torch.autograd.Variable(sample_batched['data'].cuda())
		label = torch.autograd.Variable(sample_batched['label'].cuda())

		output = model(inp.view(inp.shape[0],1,16000))

		#Loss
		loss = criterion(output, label)
		loss.backward()
		optimizer.step()

		elapsed = time.time() - t0

		if i%400 ==0:
			print('Epoch: [{0}][{1}/{2}]\t'
				'Loss {loss:.3f}\t'
				'elapsed Time: {elapsed:.2f}s'.format(e, i, N, loss=loss,elapsed=elapsed))

	scheduler.step()

	correct_count = 0
	all_count = 0
	t1 = time.time()

	for j, val_samples in enumerate(val_loader):
		inp_v = torch.autograd.Variable(val_samples['data'].cuda())
		label_v = torch.autograd.Variable(val_samples['label'].cuda())
		
		with torch.no_grad():
			model.eval()
			logps = model(inp_v.view(inp_v.shape[0],1,16000)).to(torch.device('cpu'))

		ps = torch.exp(logps)

		for j in range(inp_v.shape[0]):
			prob = list(ps.numpy()[j])
			pred_label = prob.index(max(prob))
			true_label = label_v[j]

			if(true_label == pred_label):
				correct_count+=1
			all_count+=1

		# ps = torch.exp(logps)
		# probab = list(ps.numpy()[0])
		# pred_label = probab.index(max(probab))
		# true_label = label_v

		# if(true_label == pred_label):
		# 	correct_count+=1
		# all_count+=1
	print('Time_val: {a:.0f}min {b:.0f}s\t'
		'Images tested: {c:.0f}\t'
		'Val. accuracy: {acc:.2f}'.format(a=(time.time()-t1)//60, b=(time.time()-t1)%60, c=all_count, acc=correct_count/all_count))

	if (e+1)%10 == 0:
		OUT_PATH = "/home/balaji5199/Desktop/KWS/epoch" + str(e+1) + ".pth"
		torch.save(model.state_dict(), OUT_PATH)
		print("Saved checkpoint {}".format(e+1))


FINAL_PATH = "/home/balaji5199/Desktop/KWS/final.pth"
torch.save(model.state_dict(), FINAL_PATH)

print("Training completed!")
	