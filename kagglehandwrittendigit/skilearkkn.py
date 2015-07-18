__author__ = 'Gary'

import pandas as pd
import numpy as np
import csv as csv
from sklearn.neighbors import KNeighborsClassifier
train_raw=pd.read_csv('train.csv',header=0)
test_raw=pd.read_csv('test.csv',header=0)
knn = KNeighborsClassifier()
train = train_raw.values
test = test_raw.values
print 'Start training'
knn.fit(train[0::,1::],train[0::,0])
print 'Start predicting'
out=knn.predict(test)
print 'Start writing!'
n,m=test.shape

ids = range(1,n+1)
predictions_file = open("out.csv","wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerows(["ImageId","Label"])
open_file_object.writerows(zip(ids,out)
predictions_file.close()
print 'All is done'