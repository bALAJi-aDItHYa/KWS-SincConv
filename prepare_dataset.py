# ----------------------------------------------------------------------------
# prepare_dataset.py
# Written by Balaji Adhithya V
# ----------------------------------------------------------------------------
import io
import os
import math
import time

import librosa
import requests
import torchaudio
import torch
import pandas as pd

#Dictionary of cat
GScmd_categ = {'unknown' : 0, '_silence_' : 1, '_background_noise_' : 1, 'yes' : 2, 'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 
				'right' : 7, 'on' : 8, 'off' : 9, 'stop' : 10, 'go' : 11}

numGScmd_categs= 12


base_dir = "/home/balaji5199/Desktop/KWS"
testWAVs = pd.read_csv(base_dir+"/google_speech_commands/testing_list.txt", sep=" ", header=None)[0].tolist()
valWAVs = pd.read_csv(base_dir+"/google_speech_commands/validation_list.txt", sep=" ", header=None)[0].tolist()

testWAVs = [os.path.join(base_dir+"/google_speech_commands/", f) for f in testWAVs if f.endswith('.wav')]
valWAVs = [os.path.join(base_dir+"/google_speech_commands/", f) for f in valWAVs if f.endswith('.wav')]

allWAVs = []
count=0

for root, dirs, files in os.walk(base_dir):
	allWAVs += [root + '/' + f for f in files if f.endswith('.wav')]
trainWAVs = list(set(allWAVs)-set(testWAVs)-set(valWAVs))


def get_cat(file, dic):
	categ = os.path.basename(os.path.dirname(file))
	return dic.get(categ,0)

testWAVlabels = [get_cat(f, GScmd_categ) for f in testWAVs]
trainWAVlabels = [get_cat(f, GScmd_categ) for f in trainWAVs]
valWAVlabels = [get_cat(f, GScmd_categ) for f in valWAVs]

bgWAVs = [trainWAVs[i] for i in range(len(trainWAVlabels)) if trainWAVlabels[i] == GScmd_categ['_background_noise_']]
bgWAVlabels = [GScmd_categ['_silence_'] for i in range(len(bgWAVs))]

# valWAVs += bgWAVs
# valWAVlabels += bgWAVlabels

#Replicating background noise/ silence among the training data
# for i in range(200):
# 	trainWAVs = trainWAVs + bgWAVs

testWAVlabelsDict     = dict(zip(testWAVs, testWAVlabels))
valWAVlabelsDict      = dict(zip(valWAVs, valWAVlabels))
trainWAVlabelsDict    = dict(zip(trainWAVs, trainWAVlabels))
bgWAVlabelsDict 	  = dict(zip(bgWAVs, bgWAVlabels))

trainInfo = {'files' : trainWAVs, 'labels' : trainWAVlabelsDict}
valInfo = {'files' : valWAVs, 'labels' : valWAVlabelsDict}
testInfo = {'files' : testWAVs, 'labels' : testWAVlabelsDict}
bgInfo=    {'files': bgWAVs,    'labels' : bgWAVlabelsDict}

gscInfo= {'train': trainInfo, 
          'val':   valInfo, 
          'test':  testInfo,
          'bg':    bgInfo}    

info= [(split, len(gscInfo[split]['files'])) for split in gscInfo.keys()] 
print(info)

key = ['val', 'test', 'train']

#-----------------------------------------------------------------------------------------------------------------------------------

import librosa
from tqdm import tqdm
import numpy as np

def Normalize_length(x, length=16000):
    #curX could be bigger or smaller than self.dim
    # if len(x) == length:
    #     X= x
        #print('Same dim')
    if len(x) > length: #bigger
        #we can choose any position in curX-self.dim
        randPos= np.random.randint(len(x)-length)
        X= x[randPos:randPos+length]
        #print('File dim bigger')
    else: #smaller
        randPos= np.random.randint(length-len(x))
        
        X= np.random.random(length)*1e-10
        
        X[randPos:randPos+len(x)]= x
        #print('File dim smaller')
    return X

# print(len(gscInfo['train']['files']))

xLL=[]
yLL=[]

train1x, train2x, train3x, train4x, train5x = [], [], [], [], []
train1y, train2y, train3y, train4y, train5y = [], [], [], [], []

train1x = gscInfo['train']['files'][0:10000]
train2x= gscInfo['train']['files'][10000:20000]
train3x = gscInfo['train']['files'][20000:30000]
train4x = gscInfo['train']['files'][30000:40000]
train5x = gscInfo['train']['files'][40000:]


# train1x = gscInfo['train']['files'][0:10]
# train2x= gscInfo['train']['files'][10:20]
# train3x = gscInfo['train']['files'][20:30]
# train4x = gscInfo['train']['files'][30:40]
# train5x = gscInfo['train']['files'][40:50]

# print( gscInfo['train']['labels'])
y = list(gscInfo['train']['labels'].values())


train1y = y[0:10000]
train2y = y[10000:20000]
train3y = y[20000:30000]
train4y = y[30000:40000]
train5y = y[40000:]

# train1y = y[0:10]
# train2y = y[10:20]
# train3y = y[20:30]
# train4y = y[30:40]
# train5y = y[40:50]

trainx = [train1x, train2x, train3x, train4x, train5x]
trainy = [train1y, train2y, train3y, train4y, train5y]

xLL =[]
yLL =[]

for i in range(0,1):
	aL = trainx[i]
	xL = []
	for fn in tqdm(aL):
		x, sr= librosa.load(fn, sr= None)
		if(len(x) != 16000):
			x= Normalize_length(x)
		xL += [x]
	xL = np.vstack(xL)
	xLL += [xL]
	# print(xL)
	print(len(xLL[0]))

	yL = trainy[i]
	yL = np.array(yL)
	yLL += [yL]
	# print(yL)
	print(len(yLL[0]))

# train1x, train2x, train3x, train4x, train5x = xLL
# train1y, train2y, train3y, train4y, train5y = yLL

x_train = xLL[0]
y_train = yLL[0]



# x_train = np.vstack(xLL)
# y_train = np.array(yLL)

# print(x_train.shape)
# print(y_train.shape)

# for s in ['train']:
# 	aL=  gscInfo[s]['files']
# 	xL= []
# 	for fn in tqdm(aL):
# 	    x, sr= librosa.load(fn, sr= None)
# 	    if(len(x) != 16000):
# 	    	x= Normalize_length(x)
# 	    xL += [x]
# 	xL= np.vstack(xL)
# 	xLL += [xL]

# 	yL=  list(gscInfo[s]['labels'].values())
# 	yL= np.array(yL)
# 	yLL += [yL]

# print("Conversion over!")


# x_val, x_test= xLL
# y_val, y_test= yLL
# x_train = xLL
# y_train = yLL

#-----------------------------------------------------------------------------------------------------------------------------------------

print("Conversion over!")

bgFiles= gscInfo['bg']['files']

def split_silence(bgFiles):

	noiseL= [librosa.load(fn, sr=None)[0] for fn in bgFiles]
       
	n=0
	silenceL= []
	for x in noiseL:
		t=0
		while (t+1)*16000 < x.size:
			x1sec= x[t*16000:(t+1)*16000]
			silenceL += [x1sec]
			t+=1
		n+=1

	return silenceL

silenceL= split_silence(bgFiles)
x_bg= silenceL= np.vstack(silenceL)
y_bg= np.ones(len(silenceL))

x_trainWithSil=  np.vstack((x_train, x_bg))
y_trainWithSil=  np.concatenate((y_train, y_bg))

# x_trainWithSil = x_train
# y_trainWithSil = y_train

# In[]
assert x_train.shape[0]        == y_train.shape[0]
# assert x_val.shape[0]          == y_val.shape[0]
# assert x_test.shape[0]         == y_test.shape[0]
assert x_trainWithSil.shape[0] == y_trainWithSil.shape[0]

x_trainWithSil= x_trainWithSil.astype('float32')
# x_test=         x_test.astype('float32')
# x_val=          x_val.astype('float32')

y_trainWithSil= y_trainWithSil.astype('int')
# y_test=         y_test.astype('int')
# y_val=          y_val.astype('int')

#-------------------------------------------------------------------------------------------------------------------------------------------

print("I'm here")

print(x_trainWithSil)
direc = "/home/balaji5199/Desktop/KWS/"
data_file_name = "gsc_data_train1.npz"
t0 = time.time()

if not os.path.isfile(direc+data_file_name):
	np.savez_compressed(
		direc+data_file_name, 
		x_trainWithSil1=    x_trainWithSil, 
		y_trainWithSil1=    y_trainWithSil,
		# x_val=      x_val,
		# y_val=      y_val,
		# x_test=     x_test, 
		# y_test=     y_test,
		)

dt= time.time()-t0
print(f'np.savez_compressed(), fn= {data_file_name}, dt(sec)= {dt:.2f}')

