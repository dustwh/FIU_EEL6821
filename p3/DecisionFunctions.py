#For the different datasets/perceptrons below, please comment before running.
import numpy as np
from sklearn.datasets import * 

class Perceptron:    
    def fit(self, X, y, n_iter=100):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        #Start by setting the weight to 1
        self.weights = np.zeros((n_features+1,))
        #generate column vector "1"
        X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        for i in range(n_iter):
            for j in range(n_samples):
                if y[j]*np.dot(self.weights, X[j, :]) <= 0:
                    self.weights += y[j]*X[j, :]
    def predict(self, X):
        if not hasattr(self, 'weights'):
            print('Please train model first.')
            return 
        n_samples = X.shape[0]
        #+1
        X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        y = np.matmul(X, self.weights)
        y = np.vectorize(lambda val: 1 if val > 0 else -1)(y)
        return y
    def score(self, X, y):
        pred_y = self.predict(X)
        return np.mean(y == pred_y)    
#Linearly separable
X, y = make_classification(
    n_features=2,
    n_classes=2,
    n_samples=200,
    n_redundant=0,
    n_clusters_per_class=1
    )
#Linearly inseparable
X, y = make_circles(n_samples=200, noise=0.03, factor=0.7)
#Classification with polynomial generating hyperplane
def polynom(indices_list, indices, a, b, p):
    indices = [*indices]
    if p == 0:
        indices_list.append(indices)
        return
    for i in range(a, b):
        indices.append(i)
        polynom(indices_list, indices, i, b, p-1)
        indices = indices[0:-1]
def polynomial_features(X, p):
    n, d = X.shape
    features = []
    for i in range(1, p+1):
        l = []
        polynom(l, [], 0, d, i)
        for indices in l:
            x = np.ones((n,))
            for idx in indices:
                x = x * X[:, idx]
            features.append(x)
    return np.stack(features, axis=1)
#Now generate an improved classifier that can handle the ring dataset above (it was linearly inseparable)
X = polynomial_features(X, 2)
