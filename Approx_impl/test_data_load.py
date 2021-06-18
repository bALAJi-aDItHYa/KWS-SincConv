import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def test_load():
	class beautify(Dataset):
		def __init__(self, sound, label):
			self.sound = sound
			self.label = label

		def __getitem__(self, idx):
			sample = {'data': self.sound[idx], 'label': self.label[idx]}
			return sample

		def __len__(self):
			return self.sound.shape[0]


	z = np.load('gsc_data.npz')

	test_data = z['x_test']
	test_labels = z['y_test']

	btest = beautify(test_data, test_labels)

	batch_size = 64
	test_loader = DataLoader(btest, batch_size, shuffle=False)

	return test_loader