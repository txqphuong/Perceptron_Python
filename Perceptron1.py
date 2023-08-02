

import numpy as np 
import pandas as pd 
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Perceptron1.py')


import numpy as np
import math


class Perceptron(object):
    #constructor
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta #là tốc độ học (learning rate) 
        self.n_iter = n_iter #số lần lặp lại của quá trình học 
        self.random_state = random_state #tạo các giá trị ngẫu nhiên -> nâng cao độ tin cậy của thuật toán
    
    #huấn luyện với input x, nhãn y
    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state) #sinh giá trị ngẫu nhiên
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) #sinh ngẫu nhiên trọng số w
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi)) #điều chỉnh trọng số
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    #tính tích vô hướng của vector trọng số và input tương ứng
    def net_input(self, X):

        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return z
    #dự đoán giá trị input
    def predict(self, X):

        return np.where(self.net_input(X) >= 0, 1, -1)
        import pandas as pd
#truyền dữ liệu
df = pd.read_csv('./iris.data.csv', header=None)
df.tail()


import matplotlib.pyplot as plt
import numpy as np


y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)


X = df.iloc[0:100, [0, 2]].values


plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')


from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):


    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])


    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')


plt.show()
ppn.net_input(X)
ppn.predict(X)