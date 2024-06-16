import pandas as pd
import numpy as np
import math

df=pd.read_csv("LifeExpectancyDataSet - Sheet1.csv")
df=df.dropna()

X = df[['Year', 'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI', 
          'under-five deaths', 'Polio', 'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness  1-19 years', 
          'thinness 5-9 years', 'Income composition of resources', 'Schooling']]
y = df['Life expectancy']


X=np.array(X)
y=np.array(y)

X=(X-X.mean(axis=0))/X.std(axis=0)

learning_rate = 0.01
n_iterations = 1000
m=X.shape[0]
n=X.shape[1]
X = np.c_[X , np.ones((m, 1))]
theta = np.zeros(n+1)

theta = np.zeros(X.shape[1])



for i in range(n_iterations):
    gradients = (1/m)*X.T @ (X @ theta - y) 
    theta = theta -learning_rate*gradients

predictions = X @ theta
error = predictions - y
mse= (1/m) * np.sum(np.square(error))
rmse = math.sqrt(mse)
print('RMSE Loss=', rmse)



