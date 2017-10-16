import numpy as np

def readCSV():
    data=np.loadtxt('D:/GitProject/DataMiningTest1/DataMiningTest1/Digit Recognizer/data/train.csv',dtype='U',delimiter=',')
    test=np.loadtxt('D:/GitProject/DataMiningTest1/DataMiningTest1/Digit Recognizer/data/test.csv',dtype='U',delimiter=',')
    trainLable=np.copy(data[1:,0])
    trainData=np.copy(data[1:,1:])
    trainNum=np.shape(trainLable)[0]
    testData=np.copy(test[1:,0:])
    testNum=np.shape(testData)[0]
    temp=np.zeros([testNum,1],dtype='int16')
    testLable=np.array(temp,dtype='U')
    return trainLable,trainData,trainNum,testData,testLable,testNum
a,b,c,d,e,f=readCSV()
print(type(a))
print(np.shape(a))
print(np.shape(b))
print(np.shape(d))
print(np.shape(e))
print(e)
#print(a[1][1])