import pandas as pd
import numpy as np
from MLR import ReqData
import sklearn
from sklearn import linear_model


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
    
    def run_all(self):
        self.connection_check()
        self.LinearRegressionModel()

    




o2=Model()
o2.run_all()