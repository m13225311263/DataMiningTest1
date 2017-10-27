
import torchvision
import torch.nn as nn
from torch.autograd import  Variable
import torch.utils.data as Data
import torch
from readFile import readCSV
import numpy as np
from CNNet import CNN
import torch.optim as op
import matplotlib.pyplot as plt

#parameters
EPOCH=1 #train the training data n times
BATCH_SIZE = 50
LR = 0.001 #learning rate
#read data and test
trainL,trainD,testD,testL=readCSV()

#change shape
trainD=torch.from_numpy(trainD.reshape(trainD.shape[0],28,28)).type(torch.FloatTensor)
testD=torch.from_numpy(testD.reshape(testD.shape[0],28,28)).type(torch.FloatTensor)
trainD=torch.unsqueeze(trainD,dim=1) # n*1*28*28
testD=torch.unsqueeze(testD,dim=1)
trainL=torch.from_numpy(trainL).type(torch.LongTensor)
testL=torch.from_numpy(testL).type(torch.LongTensor)

#set test
test_x=Variable(testD)
test_y=Variable(testL)
#test_y=torch.from_numpy(np.zeros((test_x.shape[0]))).type(torch.LongTensor)

#set DataLoder
torch_dataset= Data.TensorDataset(data_tensor=trainD,target_tensor=trainL)
train_loader=Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

#create cnn net

cnn=CNN()

#optimize and loss function
optimizer=op.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()


# following function (plot_with_labels) is for visualization, can be ignored if not interested
'''
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)
plt.ion()
'''






# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
#print 10 predictions from test data

test_output,x=cnn(test_x[:10])
pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')


