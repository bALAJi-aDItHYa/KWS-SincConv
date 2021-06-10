# ----------------------------------------------------------------------------
# load_data.py
# Written by Balaji Adhithya V
# 
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def data_ret():
	# Class used for preparing data for DataLoader
	class beautify(Dataset):
		def __init__(self, sound, label):
			self.sound = sound
			self.label = label

		def __getitem__(self, idx):
			sample = {'data': self.sound[idx], 'label': self.label[idx]}
			return sample

		def __len__(self):
			return self.sound.shape[0]

	# Split up training audio files in the .npz format for quick load from disk 
	train_list = ['gsc_data_train1.npz', 'gsc_data_train2.npz', 'gsc_data_train3.npz', 'gsc_data_train4.npz', 'gsc_data_train5.npz']
	train_data = []
	train_labels = []

	# -------------------------------------- Train Data ------------------------------------------
	for i in range(5):
		f = np.load(train_list[i])
		train_data += [f['x_trainWithSil'+str(i+1)]]
		train_labels += [f['y_trainWithSil'+str(i+1)]]
		print(len(train_data))

	train_data = np.vstack(train_data)
	train_labels = np.hstack(train_labels)

	# --------------------------------------------------------------------------------------------

	# -------------------------------Test and Validation data ------------------------------------

	z = np.load('gsc_data.npz')

	test_data = z['x_test']
	test_labels = z['y_test']

	val_data = z['x_val']
	val_labels = z['y_val']
	# --------------------------------------------------------------------------------------------

	# ----------------------------- Class split to find class imbalance ---------------------------

	class_split_train = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0}

	for i in train_labels:
		class_split_train[i] += 1	

	class_sample_count = []
	for i in class_split_train.keys():
		class_sample_count += [class_split_train[i]]

	class_sample_count = torch.tensor(class_sample_count)
	weights = 1./class_sample_count.float()
	sample_weights = torch.tensor([weights[i] for i in train_labels])

	sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

	# ------------------------------------------------------------------------------------------

	btrain = beautify(train_data, train_labels)
	bval = beautify(val_data, val_labels)
	btest = beautify(test_data, test_labels)

	batch_size = 16

	train_loader = DataLoader(btrain, batch_size,sampler=sampler, shuffle=False)
	val_loader = DataLoader(bval, batch_size, shuffle=True)
	test_loader = DataLoader(btest, batch_size, shuffle=False)

	return train_loader, val_loader, test_loader

