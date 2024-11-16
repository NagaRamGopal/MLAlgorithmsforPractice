import pandas as pd
import numpy as np
from MLR import ReqData
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

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
        self.y_test=y_test
        self.y_pred=model.predict(x_test)
        
        #pdt=model.predict([[324,107,8.87]]) #sample prediction
        #print(pdt)

    def metrics_evaluation(self):
        rscore=r2_score(self.y_test, self.y_pred)
        mse=mean_squared_error(self.y_test,self.y_pred)
        print("R2Score: ",rscore)
        print("Mean Squared Error: ",mse)

    def visualization_predictions(self):
        plt.figure()
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.scatter(self.y_test,self.y_pred, color='Blue')
        plt.show()


    
    
    def run_all(self):
        self.connection_check()
        self.LinearRegressionModel()
        self.metrics_evaluation()
        self.visualization_predictions()

    
    




o2=Model()
o2.run_all()