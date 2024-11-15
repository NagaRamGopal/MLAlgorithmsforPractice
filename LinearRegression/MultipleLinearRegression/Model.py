import pandas as pd
import numpy as np
from MLR import ReqData
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split


class Model:
    def connection_check(self):
        self.df=ReqData
        if self.df.empty:
            print("Data Not Received")
        else:
            print("Connection Successfull")
            print("Data Received")
            print(ReqData.head())

    def LinearRegressionModel(self):
        model=linear_model.LinearRegression()
        x = self.df[['GRE Score', 'TOEFL Score', 'CGPA']]
        y=self.df['Chance of Admit']
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=22) #setting random state to select same data everytime for training/testing.
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        pdt=model.predict([[324,107,8.87]]) #sample prediction
        print(pdt)
    
    def run_all(self):
        self.connection_check()
        self.LinearRegressionModel()

    




o2=Model()
o2.run_all()