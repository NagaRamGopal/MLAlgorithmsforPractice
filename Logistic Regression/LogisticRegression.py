import pandas as pd
import numpy as np
import pandas_profiling
from pandas_profiling import ProfileReport

class LR:

    @staticmethod
    def read_data():
        LR.df=pd.read_csv('C:/Users/ramgo/OneDrive/Desktop/Learn/MLAlgorithmsforPractice/Logistic Regression/bankloan.csv')
        if LR.df.empty:
            print("Data loading error")
        else:
            print(LR.df.head())

    @staticmethod
    def data_report():
        rpt=ProfileReport(LR.df,title="Report")
        rpt.to_file("Report.html")

    @staticmethod
    def run_all():
        LR.read_data()
        LR.data_report()

obj1=LR()
obj1.run_all()