import pandas as pd
import numpy as np
import pandas_profiling
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

class MLR:
    def read_data(self):
        try:
            self.df=pd.read_csv(r'C:\Users\ramgo\OneDrive\Desktop\Learn\MLAlgorithmsforPractice\LinearRegression\MultipleLinearRegression\jamboree_dataset.csv')
        except:
            print("Data Loading error")

    def statistics_data(self):
        rpt=ProfileReport(self.df,title="Data Report before EDA")
        rpt.to_file("DataReport.html")
        
        
    def cleaning_data(self):   #From report we can see no duplicates, no missing values. 
        self.df.drop(['Serial No.'],axis=1,inplace=True)
        print(self.df.corr())
        sns.heatmap(self.df.corr(),cmap='coolwarm')
        plt.title('Correlation')
         #we can see chance of admit is highly corelated to gre, toefl scores and @same time those who got gre more=toefl more

    def bivariate_analysis(self):
      plt.figure()
      sns.scatterplot(x=self.df['GRE Score'], y=self.df['TOEFL Score'])
      plt.title('GRE vs TOEFL Score')
      plt.show()

      plt.figure()
      sns.scatterplot(x=self.df['GRE Score'], y=self.df['CGPA'])
      plt.title('GRE vs CGPA')
      plt.show()

      plt.figure()
      sns.scatterplot(x=self.df['TOEFL Score'], y=self.df['CGPA'])
      plt.title('TOEFL Score vs CGPA')
      plt.show()


    def final_columns(self):
        self.df.drop(['SOP', 'LOR ', 'University Rating', 'Research'], axis=1, inplace=True)
        print(self.df.columns)
        global ReqData
        self.df.columns = self.df.columns.str.strip()  #to remove space at the end. 
        print(self.df.columns)
        ReqData=self.df
        

    def run_all(self): #we can see that as gre increases, toefl increases too.
        self.read_data()
        self.statistics_data()
        self.cleaning_data()
        self.bivariate_analysis()
        self.final_columns()

        


o1=MLR()
o1.run_all()