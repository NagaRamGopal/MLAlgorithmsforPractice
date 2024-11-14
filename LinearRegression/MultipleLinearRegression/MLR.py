import pandas as pd
import numpy as np
import pandas_profiling
from pandas_profiling import ProfileReport

class MLR:
    def read_data(self):
        try:
            self.df=pd.read_csv(r'C:\Users\ramgo\OneDrive\Desktop\LinearRegressionPractice\MultipleLinearRegression\jamboree_dataset.csv')
            #print(self.df.head())
        except:
            print("Data Loading error")

    def statistics_data(self):
        rpt=ProfileReport(self.df,title="Data Report before EDA")
        rpt.to_file("DataReport.html")
        
        
    def cleaning_data(self):   #From report we can see no duplicates, no missing values. 
        self.df.drop(['Serial No.'],axis=1,inplace=True)
        #print(self.df.head())


    def data_visualization(self):
        pass

    def run_all(self):
        self.read_data()
        #self.statistics_data()
        self.cleaning_data()
        self.data_visualization()


o1=MLR()
o1.run_all()