import numpy as np


def readCSV():
    data=np.loadtxt('../data/train.csv',dtype='U',delimiter=',')
    test=np.loadtxt('../data/test.csv',dtype='U',delimiter=',')
    #data.dtype='int16'
    #test.dtype='int16'
    trainLable=np.copy(data[1:,0])
    trainData=np.copy(data[1:,1:])
    trainNum=np.shape(trainLable)[0]
    testData=np.copy(test[1:,0:])
    testNum=np.shape(testData)[0]
    trainData=toInt(trainData)
    trainLable=toInt(trainLable)
    testData=toInt(testData)
    temp=np.zeros([testNum,1],dtype='int16')
    testLable=np.array(temp,dtype='int16')
    return trainLable,trainData,testData

#change the string to int and normalization
def toInt(array):
    array=np.mat(array)
    m=array.shape[0]
    n=array.shape[1]
    newA=np.zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            newA[i,j]=float(array[i,j])/255 # make sure the range of data in (0,1)
    return newA
'''
a,b,c,d,e,f=readCSV()
print(type(a))
print(np.shape(a))
print(np.shape(b))
print(np.shape(d))
print(np.shape(e))
print(e)
#print(a[1][1])
'''
