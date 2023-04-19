# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:45:58 2023

@author: niawi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from numpy import NaN

df_data = pd.read_csv("C:/Users/niawi/Onedrive/Documents/UCL/qmyear2/spyderdata/Reg_data_pm25/2021_pm25.csv")

#df_data = df_data[np.random.permutation(df_data.columns)] #this randomises the position of the columns
df_data['PM25 level'] = np.random.permutation(df_data['PM25 level'].values) #this shuffles the data within a specified column

df_data.dropna(inplace=True)

df_data.drop('Year', axis=1, inplace=True)

print(df_data)


y = df_data.iloc[:, 1:].values
x = df_data.iloc[:, :1].values

regressor = LinearRegression()
regressor.fit(x, y)
model = sm.OLS(y, x, missing='drop')
results = model.fit()

plt.scatter(x, y, color = 'green')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Relationship between happiness and levels \n of PM25 in different locations in 2021')
plt.xlabel('PM25 level')
plt.ylabel('Happiness')


#define response variable
y = df_data['Happiness']

#define predictor variables
x = df_data[['PM25 level']]

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())

plt.savefig('2021_pm25.jpg', bbox_inches='tight')
plt.show()
