# ----------------------------------------------------------------------------
# test.py
# Written by Balaji Adhithya V
# ----------------------------------------------------------------------------
import torch
from SincNet_Model import SincNet
from load_data import data_ret
import time

model = SincNet().cuda()
PATH = "/home/balaji5199/Desktop/KWS/final.pth"
model.load_state_dict(torch.load(PATH))

train_loader, val_loader, test_loader = data_ret()

correct_count = 0
all_count= 0

print("Calculating model accuracy....")
t0 = time.time()

for i, samples in enumerate(test_loader):

	inp = torch.autograd.Variable(samples['data'].cuda())
	label = torch.autograd.Variable(samples['label'].cuda())
	device = torch.device('cpu')

	with torch.no_grad():
		model.eval()
		output = model(inp.view(inp.shape[0],1,16000)).to(device)

	ps = torch.exp(output)

	for j in range(inp.shape[0]):
		prob = list(ps.numpy()[j])
		pred_label = prob.index(max(prob))
		true_label = label[j]

		if(true_label == pred_label):
			correct_count+=1
		all_count+=1

end = time.time()-t0
acc = correct_count/all_count

print('Time: {a:.0f}min {b:.0f}s\n'
	  'Images tested: {all:.0f}\n'
	  'test accuracy: {acc:.3f}\n'.format(
	  	a=end/60, b=end%60,
	  	all = all_count, 
	  	acc = acc))
