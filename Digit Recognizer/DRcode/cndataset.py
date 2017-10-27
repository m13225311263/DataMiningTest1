from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
import torch
import  pandas as pd


class HandwriteDataset(Dataset):
    def __init__(self,csv_file, root_dir, transform=None):

        self.landmarks_frame=np.loadtxt(csv_file,dtype='U',delimiter=',')
        self.landmarks_frame=self.landmarks_frame[1:,:]
        self.root_dir=root_dir
        self.transform=transform


    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        self.datesets= self.landmarks_frame[idx,1:]
        lable=self.landmarks_frame[idx,0]
        sample={'dataitem':self.datesets,'label':lable}
        return sample

    def toInt(array):
        array = np.mat(array)
        m = array.shape[0]
        n = array.shape[1]
        newA = np.zeros((m, n))
        for i in xrange(m):
            for j in xrange(n):
                newA[i, j] = float(array[i, j]) / 255  # make sure the range of data in (0,1)
        return newA

c=HandwriteDataset(csv_file='../data/train.csv',root_dir='../data/')
c[1]['label']
train_loader=DataLoader(datasets=c,batch_size=2,shuffle=True)






