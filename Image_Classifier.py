#Import the required librabries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline

#Read train data
data = pd.read_csv('mnist_train.csv')
data.head()

#Visualize the dataset
a = data.iloc[4,1:].values
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)

#Seperate the train data as data and label
train_data_x=data.iloc[:,1:]
train_data_label=data.iloc[:,0]
train_data_x.head()

#Read test data and seperate the data and the label
test_data=pd.read_csv('mnist_test.csv')
test_data_x=test_data.iloc[:,1:]
test_data_label=test_data.iloc[:,0]
test_data_x.head()

#Call RF Classifier and train the model
rf=RandomForestClassifier(n_estimators=100)
rf.fit(train_data_x,train_data_label)

#Predicting with test_data
pred=rf.predict(test_data_x)
pred

#Calculate the accuracy
test=test_data_label.values
count=0

for i in range(len(pred)):
    if(pred[i]==test[i]):
        count=count+1
        
accuracy=(count/len(pred))*100
print(accuracy,"%")
