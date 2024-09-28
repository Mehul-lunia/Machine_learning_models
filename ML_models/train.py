import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from  Linear_regression import LinearRegression

X,y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0],y,color='b',marker='o',s=30)
plt.show()

# fitting the data
model = LinearRegression(lr=0.1)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

def mean_sqaured_error():
    return np.mean((y_pred-y_test)**2)
mse = mean_sqaured_error()
print(mse)

y_pred_line = model.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train,y_train,color=cmap(0.9),s=10)
m1 = plt.scatter(X_test,y_test,color=cmap(0.5),s=10)
plt.plot(X,y_pred_line,color="black",linewidth=2,label="predictions")
plt.show()