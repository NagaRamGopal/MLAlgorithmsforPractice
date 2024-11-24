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
        col=LR.df.columns
        for c in col:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=LR.df[c])
            plt.title(f"Boxplot for {c}")
            plt.show() #we can see outliers for income, mortgage and ccavg columns
    
    @staticmethod
    def handling_outliers():
        col_req=LR.df[['Income', 'CCAvg', 'Mortgage']]
        for i in col_req:
            Q1=LR.df[i].quantile(0.25)
            Q3=LR.df[i].quantile(0.75)
            IQR=Q3-Q1
            lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
            upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
            LR.df = LR.df[(LR.df[i] >= lower_bound) & (LR.df[i] <= upper_bound)]
        LR.df.dropna(inplace=True)
        print(LR.df.shape)
    
    
    @staticmethod
    def skewness_check():
        skewcol=LR.df[['Income','CCAvg','Mortgage']]
        skewness = skewcol.skew()
        print("skewness before", skewness) #we see we are having +ve skewness
        
    
    @staticmethod
    def handling_skewness():
        skewcol=['Income','CCAvg','Mortgage']
        for i in skewcol:
            LR.df[i]=np.sqrt(LR.df[i])
        after_handle_skewness=LR.df[skewcol].skew()
        print("Handled skewness with Square Root Transformation ",after_handle_skewness)
        #LR.df['Mortgage'] = np.log1p(LR.df['Mortgage']) 
        #print("Updated Mortgage Skewness:", LR.df['Mortgage'].skew())

    @staticmethod
    def correlation():
        plt.figure(figsize=(12,4))
        sns.heatmap(LR.df.corr(),annot=True)
        plt.show()
        #print(LR.df.corr())




    @staticmethod
    def run_all():
        LR.read_data()
        LR.data_report()
        LR.cleaning_data()
        LR.outliers_detection()
        LR.handling_outliers()
        LR.skewness_check()
        LR.handling_skewness()
        LR.correlation()


obj1=LR()
obj1.run_all()