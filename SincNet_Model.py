# -----------------------------------------------------------------------
# SincNet_Model.py
# Written by Balaji Adithya V
# -----------------------------------------------------------------------
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

def flip(x, dim):
	xsize = x.size()
	dim = x.dim() + dim if dim < 0 else dim
	x = x.contiguous()
	x = x.view(-1, *xsize[dim:])
	x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
	                  -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
	return x.view(xsize)

def sinc(band,t_right):
	y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
	y_left= flip(y_right,0)

	y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

	return y

class log_compression(nn.Module):
	def __init__(
		self,
		num_features: int):
		super(log_compression, self).__init__()
		self.bnorm = nn.BatchNorm1d(num_features)
		self.avgpool = nn.AvgPool1d(2)
	
	def forward(self, x):
		#log - compression (used to scale the range of values from 0-n)
		x = torch.add(torch.abs(x),1)
		x = torch.log2(x)

		x = self.bnorm(x)
		x = self.avgpool(x)
		return x

class DS_conv_op(nn.Sequential):
	def __init__(
		self,
		input_channels: int, 
		output_channels: int, 
		kernel_size: int, 
		stride: int, 
		groups: int
		) -> None:

		super(DS_conv_op, self).__init__(
			nn.Conv1d(input_channels, input_channels, kernel_size = kernel_size, stride=stride, groups=input_channels), #Depthwise
			nn.Conv1d(input_channels, output_channels, kernel_size =1, stride=stride), #Pointwise
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(output_channels),
			nn.AvgPool1d(kernel_size=2, stride=None, padding=1)
			)


class DSConv_block(nn.Module):
	def __init__(
		self, 
		input_channels: int,
		output_channels: int,
		kernel_size: int,
		stride: int
		) -> None:
		super(DSConv_block, self).__init__()
		self.stride = stride
		self.kernel_size = kernel_size

		ds_layers: List[nn.Module] = []
		ds_layers.append(DS_conv_op(input_channels, output_channels, self.kernel_size, self.stride, groups=input_channels))
		
		self.ds_layers = nn.Sequential(*ds_layers)
		self.drop = nn.Dropout(0.1)	

	def forward(self, x):
		x = self.ds_layers(x)
		# print("In DS_conv_op")
		# print(x.shape)
		x = self.drop(x)

		return x



class sinc_conv(nn.Module):
	def __init__(self, n_filt=40, filt_dim=101,fs=16000):
		super(sinc_conv, self).__init__()

		low_freq_mel = 80
		high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
		mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filt)  # Equally spaced in Mel scale
		f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
		b1=np.roll(f_cos,1)
		b2=np.roll(f_cos,-1)
		b1[0]=30
		b2[-1]=(fs/2)-100
		        
		self.freq_scale=fs*1.0
		self.filt_b1 = nn.Parameter(torch.from_numpy(b1/self.freq_scale))
		self.filt_band = nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))

		self.n_filt=n_filt
		self.filt_dim=filt_dim
		self.fs=fs


	def forward(self, x):
		filters=Variable(torch.zeros((self.n_filt,self.filt_dim))).cuda()
		N=self.filt_dim
		t_right=Variable(torch.linspace(1, (N-1)/2, steps=int((N-1)/2))/self.fs).cuda()


		min_freq=50.0;
		min_band=50.0;

		filt_beg_freq=torch.abs(self.filt_b1)+min_freq/self.freq_scale
		filt_end_freq=filt_beg_freq+(torch.abs(self.filt_band)+min_band/self.freq_scale)

		n=torch.linspace(0, N, steps=N)

		# Filter window (hamming)
		window=0.54-0.46*torch.cos(2*math.pi*n/N);
		window=Variable(window.float().cuda())

        
		for i in range(self.n_filt):
		                
		    low_pass1 = 2*filt_beg_freq[i].float()*sinc(filt_beg_freq[i].float()*self.freq_scale,t_right)
		    low_pass2 = 2*filt_end_freq[i].float()*sinc(filt_end_freq[i].float()*self.freq_scale,t_right)
		    band_pass=(low_pass2-low_pass1)

		    band_pass=band_pass/torch.max(band_pass)

		    filters[i,:]=band_pass.cuda()*window

		# print(filters.view(40,1,101).shape)

		out = F.conv1d(x, filters.view(self.n_filt,1,self.filt_dim), stride=8)
		# print("out in sinc")
		# print(out.shape)

		return out


class SincNet(nn.Module):
	def __init__(
		self,
		num_classes: int = 12,
		mode: int = 0
		) -> None:

		super(SincNet, self).__init__()
		setting = [

			#c 	#k	#s
			[160, 25, 2],
			[160, 9 ,1],
			[160, 9 ,1],
			[160, 9 ,1],
			[160, 9 ,1],					
		]
		n_filt = 40 #Output channels from SincConv
		filt_dim = 101 #Kernel length for sincconv-layer
		fs = 16000 #sampling rate of the audio file
	
		sincconv_block: List[nn.Module] = [sinc_conv(n_filt, filt_dim, fs)]
		sincconv_block.append(log_compression(n_filt))

		self.sincconv_block = nn.Sequential(*sincconv_block)
		
		
		# ds_features: List[nn.Module] = [DSConv_block(n_filt, output_channels=160, kernel_size=25, stride=2)]
		

		ds_features: List[nn.Module] = []
		input_channels = n_filt
		for c,k,s in setting:
			# print(k)
			output_channels = c
			ds_features.append(DSConv_block(input_channels= input_channels, output_channels=c, kernel_size=k, stride=s))
			input_channels = output_channels
		
		self.ds_features = nn.Sequential(*ds_features)
		self.global_avg = nn.AdaptiveAvgPool1d(1)

		self.classifier = nn.Sequential(
			nn.Linear(160, num_classes),
			nn.LogSoftmax(dim=1)
			)

	def forward(self, x: Tensor)-> Tensor:
		x = self.sincconv_block(x)
		# print(x.shape)
		x = self.ds_features(x)
		x = self.global_avg(x)
		x = torch.flatten(x,1)
		# print(x.shape)		
		x = self.classifier(x)

		return x