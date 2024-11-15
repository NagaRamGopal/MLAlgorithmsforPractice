import pandas as pd
import numpy as np
from MLR import ReqData


class Model:
    def connection_check(self):
        self.df=ReqData
        if self.df.empty:
            print("Data Not Received")
        else:
            print("Connection Successfull")
            print("Data Received")
            print(ReqData.head())
    
    def run_all(self):
        self.connection_check()



o2=Model()
o2.run_all()