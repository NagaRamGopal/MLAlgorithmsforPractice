import pandas as pd
import numpy as np
import pandas_profiling
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import sys


class LR:

    @staticmethod
    def read_data():
        LR.df=pd.read_csv('C:/Users/ramgo/OneDrive/Desktop/Learn/MLAlgorithmsforPractice/Logistic Regression/bankloan.csv')
        if LR.df.empty:
            print("Data loading error")
            sys.exit()

    @staticmethod
    def data_report():
        rpt=ProfileReport(LR.df,title="Report")
        rpt.to_file("Report.html")

    @staticmethod
    def cleaning_data():
        LR.df.drop(['ID','ZIP.Code'],axis=1,inplace=True)
        LR.df.columns=LR.df.columns.str.strip()
        checknegvalues=LR.df.lt(0).sum() #gives count of values that are less than zero from columns
        print(checknegvalues) # we can see experience has 52 values<0. usually experience can't be less than 0. 
        LR.df['Experience'] = abs(LR.df['Experience']) #now all values from experience will be greater than 0 and +ve.
        print(LR.df[LR.df['Experience']<0]) #cross checking again to see if there any values left from experience column.
        #print(LR.df[LR.df.duplicated()]) #repeated rows
        print(LR.df.duplicated().sum())
        LR.df.drop_duplicates(inplace=True)
        print(LR.df.head())
        #print(LR.df.duplicated().sum()) #cross checking again to see repeated rows are left.
        #print(LR.df.isnull().sum())  #checking any missing values.

    @staticmethod
    def outliers_detection():
        col=LR.df[['Age','Experience','Income','CCAvg']]
        for c in col:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=LR.df[c])
            plt.title(f"Boxplot for {c}")
            plt.show()
    
    @staticmethod
    def statistics_check():
        print(LR.df['Age'].mean().round(), LR.df['Age'].median().round(), LR.df['Age'].mode().round())


    @staticmethod
    def run_all():
        LR.read_data()
        #LR.data_report()
        LR.cleaning_data()
        LR.outliers_detection()
        LR.statistics_check()
    


obj1=LR()
obj1.run_all()