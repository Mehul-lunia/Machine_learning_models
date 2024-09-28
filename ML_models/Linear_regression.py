import numpy as np

class LinearRegression:
    def __init__(self,lr=0.001,n_iter=1000) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        # get the number of features and create a weight array of same size
        num_of_rows,num_of_features = X.shape
        self.weights = np.zeros(num_of_features)
        self.bias = 0
        # We want to predict for all the rows/samples at the same time
        for _ in range(self.n_iter):
            y_pred = np.dot(self.weights,X.T) + self.bias

            dw = (1/num_of_rows)*np.dot(X.T,(y_pred-y))
            db = (1/num_of_rows)*np.sum(y_pred-y)

            self.weights = self.weights - (self.lr*dw)
            self.bias = self.bias - (self.lr*db)

    def predict(self,X):
        y_pred = np.dot(self.weights,X.T) + self.bias
        return y_pred

        