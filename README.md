# KWS-SincConv #
Applying approximations in the models aimed at the task of Key Word Spotting.

## Dataset used ##
The dataset used for the project is [Google Speech Commands dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)
The dataset contains 62K images which can be split into train, test and validation splits based on the txt files provided.

## Training ##
The basic steps involved are:

###	1) Preparing the dataset ###
1) The dataset requires some clean up before it can directly be used as input to the model as the audio files are all not sampled at the same rate
2) The _prepare_dataset.py_ is used to normalise the lengths of the inputs and convert them to .npz format for easy loading onto the training step.
3) The test, train, validate split is made based on the _testingList.txt_ and _validationList.txt_ as provided by the authors of the dataset.
