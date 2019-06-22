# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:07:16 2019

@author: Shrunali
"""

# Importing the Necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
datas=pd.read_csv('Salary_Data.csv')

#splitting the dataset according to dependent and independent variables.

x=datas.iloc[:,0:1].values #independent
y=datas.iloc[:,1].values #dependent

#to split data as train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#applying the linear regression model i.e fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(x_train,y_train)

#predicting the data
xpredict=LR.predict(x_test)

#visualizing the training set results
plt.scatter(x_train,y_train,edgecolors='blue')
plt.plot(x_train,LR.predict(x_train),color='red')
plt.title('prediction of year of experience vs salary')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

 #visualizing the test set results
plt.scatter(x_test,y_test,edgecolors='blue')
plt.plot(x_train,LR.predict(x_train),color='red')
plt.title('prediction of year of experience vs salary')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()