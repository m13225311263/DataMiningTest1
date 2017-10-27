
import torchvision
import torch.nn as nn
from torch.autograd import  Variable
import torch
from readFile import readCSV
import numpy as np

#read data and test
trainL,trainD,testD=readCSV()

#change shape
trainData=np.array([torch.from_numpy(trainD[i].reshape(28,28))for i in xrange (trainD.shape[0])])
testData=np.array([ torch.from_numpy(testD[i].reshape(28,28)) for i in xrange(testD.shape[0])])

# numpy->torch
#dataD=[torch.from_numpy(trainData[i]) for i in xrange (trainData.shape[0])]
#testD=torch.from_numpy(testData[i] for i in xrange (testData.shape[1]))
trainLable=np.array([torch.from_numpy(np.array(int(trainL[i]))) for i  in xrange(trainL.shape[0])])

torch.from_numpy(trainL)
print(trainD.shape)

