
import numpy as np
import torch
from readFile import readCSV
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from CNNet import CNN
from torch.optim import Adam
import torch.nn as nn

#parameters
EPOCH=1 #train the training data n times
BATCH_SIZE = 50
LR = 0.001 #learning rate

trainLable,trainData,testData=readCSV()
trainData=[torch.from_numpy(trainData[i].reshape(28,28))for i in xrange (trainData.shape[0])]
testData=[ torch.from_numpy(testData[i].reshape(28,28)) for i in xrange(testData.shape[0])]

#trainData=np.array(trainData)
#testData=np.array(testData)

train_loader=DataLoader(datasets=trainData,batch_size=2,shuffle=True)


#numpy-> tensor
#trT=torch.from_numpy(trainLable).type(torch.FloatTensor)
#trD=torch.from_numpy(trainData).type(torch.FloatTensor)
#testD=torch.from_numpy(testData).type(torch.FloatTensor)
#Tlable,TData=Variable(trT),Variable(trD)

#create dataloader
#train_loader=DataLoader(dataset=trD,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

#create net
cnn=CNN()
# print(cnn)
optimizer=Adam(cnn.parameters(),lr=LR) # optimize all cnn parameters
loss_func=nn.CrossEntropyLoss()

# training
for epoch in xrange(EPOCH):
    for i in xrange(1,trD):
        b_x=Variable(np.reshape(trD[i],(1,28,28)))
        b_y=Variable(trT[i])
        output=cnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i %50 ==0:
            test_output=cnn()
            pred_y=torch.max(test_output,1)[1]
            accuracy=sum(testD==testT)/testT.shape()[0]
            print('Epoch:',epoch,'| train loss: %.4f' % loss.data[0],' ')



'''
trainx,trainy,=Variable(trD),Variable(trT)


'''
#design net