import pandas as pd
import numpy as np
import csv as csv
import kNN

from sklearn.neighbors import KNeighborsClassifier

train_raw=pd.read_csv('train.csv',header=0)
test_raw=pd.read_csv('test.csv',header=0)

train=train_raw.values
test=test_raw.values


labels=train[0::,0:1]
labels=labels[:,0]

i,j=train.shape

result=[]

for ii in range(i):
    for jj in range(j):
         if (train[ii,jj]>1):
             train[ii,jj]=1


for iii in range(i):
    result[iii,0]=kNN.classify0(train[iii:iii+1,0::],train,labels,4)

predictions_file = open("out.csv","wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerows(["Label"])
predictions_file.close()





