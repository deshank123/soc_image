import pandas as pd
import numpy as np
import math

df=pd.read_csv("framingham.csv")
df=df.dropna()
y=df["TenYearCHD"]
X=df.drop(columns=["TenYearCHD"])
X=np.array(X)
y=np.array(y)
m=len(y)

X=(X-X.mean(axis=0))/X.std(axis=0)


X = np.hstack([np.ones((X.shape[0], 1)), X])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    
    for i in range(num_iterations):
        gradient = (1/m) * X.T.dot(sigmoid(X.dot(theta)) - y)
        theta = theta - learning_rate * gradient
        
    return theta

theta = np.zeros(X.shape[1])
learning_rate = 0.02
num_iterations = 2000

theta = gradient_descent(X, y, theta, learning_rate, num_iterations)


def predict(X, theta):
    return (sigmoid(X.dot(theta)) >= 0.5).astype(int)

# Evaluate the model
predictions = predict(X, theta)
accuracy = np.mean(predictions == y) * 100
print(f'Accuracy: {accuracy:.2f}%')




false_pos=0
false_neg=0
pos=0
neg=0
pred_neg=0
pred_pos=0
for i in range(m):
    if y[i]==1:
        pos+=1
        if sigmoid(X[i].dot(theta)) >= 0.5:
            false_neg+=1
    else:
        neg+=1
        if sigmoid(X[i].dot(theta)) >= 0.5:
            false_pos+=1

for i in range(m):
    if sigmoid(X[i].dot(theta)) >= 0.5:
        pred_neg+=1
    else:
        pred_pos+=1
fal_pos = false_pos*100.0/neg
fal_neg = false_neg*100.0/pos

print('False positive=', fal_pos, '%')
print('False negative=', fal_neg, '%')
