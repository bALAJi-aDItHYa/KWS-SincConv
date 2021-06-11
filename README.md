# KWS-SincConv #
Applying approximations in the models aimed at the task of Key Word Spotting.

## Dataset used ##
The dataset used for the project is [Google Speech Commands dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)
The dataset contains 62K images which can be split into train, test and validation splits based on the txt files provided.

## Steps to replicate ##
The basic steps involved are:

### 1) Preparing the dataset ###
1) The dataset requires some clean up before it can directly be used as input to the model as the audio files are all not sampled at the same rate
2) The _prepare_dataset.py_ is used to normalise the lengths of the inputs and convert them to .npz format for easy loading onto the training step.
3) The test, train, validate split is made based on the _testingList.txt_ and _validationList.txt_ as provided by the authors of the dataset.

### 2) Model Arcitecture ###
1) The model that I am trying to emulate is from [Simon et al.](https://arxiv.org/pdf/1911.02086.pdf)
2) The name of the model is SincNet that utilizes the Sinc function as filter for the first in order to circumvent the use of pre-processing, such as MFCC, on input data.
3) The model description is in _SincNet\_Model.py_
4) The implementation of SincConv layer has been used from [here](https://github.com/mravanelli/SincNet)

### 3) Training on GPU ###
1) The training was done on a GTX 1050Ti - 4GB RAM. 
2) The model was trained for 60 epochs, with an initial learning rate of 0.001 and a lr decay by 0.5 every 10 epochs.
3) Adam Optimizer used.
4) Trained on PyTorch!!

The testing accuracy after the following steps is - __94.65%__ over an test-dataset of 6835 audio files
The number of parameters are - 

              Modules                           | Parameters 
| :---: | :---: | 
sincconv_block.0.filt_b1      			|40
sincconv_block.0.filt_band    			|40
sincconv_block.1.bnorm.weight    		|40
sincconv_block.1.bnorm.bias     		|40
ds\_features.0.ds_layers.0.0.weight 		|1000
ds\_features.0.ds_layers.0.0.bias  		|40
ds\_features.0.ds_layers.0.1.weight	 	|6400
ds\_features.0.ds_layers.0.1.bias  		|160
ds\_features.0.ds_layers.0.3.weight 		|160
ds\_features.0.ds_layers.0.3.bias  		|160
ds\_features.1.ds_layers.0.0.weight 		|1440
ds\_features.1.ds_layers.0.0.bias  		|160
ds\_features.1.ds_layers.0.1.weight 		|25600
ds\_features.1.ds_layers.0.1.bias  		|160
ds\_features.1.ds_layers.0.3.weight 		|160
ds\_features.1.ds_layers.0.3.bias  		|160
ds\_features.2.ds_layers.0.0.weight 		|1440
ds\_features.2.ds_layers.0.0.bias  		|160
ds\_features.2.ds_layers.0.1.weight 		|25600
ds\_features.2.ds_layers.0.1.bias  		|160
ds\_features.2.ds_layers.0.3.weight 		|160
ds\_features.2.ds_layers.0.3.bias  		|160
ds\_features.3.ds_layers.0.0.weight   		|1440
ds\_features.3.ds_layers.0.0.bias       	|160
ds\_features.3.ds_layers.0.1.weight   		|25600
ds\_features.3.ds_layers.0.1.bias       	|160 
ds\_features.3.ds_layers.0.3.weight   		|160 
ds\_features.3.ds_layers.0.3.bias       	|160 
ds\_features.4.ds_layers.0.0.weight   		|1440
ds\_features.4.ds_layers.0.0.bias       	|160 
ds\_features.4.ds_layers.0.1.weight    		|25600
ds\_features.4.ds_layers.0.1.bias       	|160 
ds\_features.4.ds_layers.0.3.weight   		|160 
ds\_features.4.ds_layers.0.3.bias       	|160 
classifier.0.weight                             |1920
classifier.0.bias                               |12 

Total Trainable Params: 120732 ~= __120K parameters__

Size of the final model on Disk - __500KB__

## Further steps ##
The next step is applying approximation in the sinc and log compression functions in the model and observe the effect on model accuracy.
Proposed approximation techniques for 
1) sinc - CORDIC + [SIMDive](https://arxiv.org/abs/2011.01148) 
2) logCompression - Not yet decided XD
