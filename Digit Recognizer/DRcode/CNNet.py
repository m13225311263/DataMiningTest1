import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d( #(1*28*28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(), #(16*28*28)
            nn.MaxPool2d(2) #(16*14*14)
        )
        self.conv2=nn.Sequential( #(16*14*14)
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),#(32*14*14)
            nn.MaxPool2d(2) #(32*7*7)
        )
        self.out=nn.Linear(32*7*7,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        # x=x.view(x.size(0),-1)
        output= self.out(x)
        return  output

cnn=CNN()
print(cnn)
