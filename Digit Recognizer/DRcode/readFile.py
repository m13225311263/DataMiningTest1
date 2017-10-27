import numpy as np


def readCSV():
    data=np.loadtxt('../data/train.csv',dtype='U',delimiter=',')
    test=np.loadtxt('../data/test.csv',dtype='U',delimiter=',')

    trainLable=np.copy(data[1:-2000,0]).astype('int32') # use 50 items to test
    trainData=np.copy(data[1:-2000,1:])

    testData=np.copy(data[-2000:,1:])
    testLable = np.copy(data[-2000:, 0]).astype('int32')

    trainData=toFloat(trainData)
    testData=toFloat(testData)

    return trainLable,trainData,testData,testLable

#change the string to int and normalization
def toFloat(array):
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
