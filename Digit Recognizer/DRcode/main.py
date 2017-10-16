import numpy as np

def readCSV():
    data=np.loadtxt('D:/GitProject/DataMiningTest1/DataMiningTest1/Digit Recognizer/data/train.csv',dtype='U',delimiter=',')
    test=np.loadtxt('D:/GitProject/DataMiningTest1/DataMiningTest1/Digit Recognizer/data/test.csv',dtype='U',delimiter=',')
    trainLable=np.copy(data[1:,0])
    trainData=np.copy(data[1:,1:])
    testData=np.copy(test[1:,1:])
    return trainLable,trainData,testData
a,b,c=readCSV()
print(type(a))
print(np.shape(a))
print(np.shape(b))
print(np.shape(c))
print(c[0])
#print(a[1][1])