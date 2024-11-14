import pandas as pd
import numpy as np
import pandas_profiling
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv(r'C:\Users\ramgo\OneDrive\Desktop\Learn\ML Algos\Linear Regression\Salary_dataset.csv')
#rpt=ProfileReport(df,title='LinearRegression')
#rpt=rpt.to_file("Report.html")
print(df.head())


x=df[['YearsExperience']].values  #Independent variable
y=df['Salary']           #Dependent variable

'''
plt.scatter(x, y)
plt.xlabel('Years of Experience')  # Label for x-axis
plt.ylabel('Salary')  # Label for y-axis
plt.title('Years of Experience vs Salary')  # Title of the plot
plt.show()
'''

model=LinearRegression()
model.fit(x,y)

y_pred = model.predict(x)

plt.scatter(x, y, color='blue')  # Actual data points
plt.plot(x, y_pred, color='red')  # Regression line
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Years of Experience vs Salary')
plt.show()

slope = model.coef_
intercept = model.intercept_
print(f"Slope: {slope}, Intercept: {intercept}")

r_squared = model.score(x, y)
print(f"R-squared: {r_squared}")

new_data = np.array([[5]])  # Predicting salary for 5 years of experience
predicted_salary = model.predict(new_data)
print(f"Predicted Salary for 5 years of experience: {predicted_salary[0]}")