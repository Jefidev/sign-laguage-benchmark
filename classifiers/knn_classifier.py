from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNNClassifier:
    def __init__(self, n_neighbors=5, distance="euclidian", n_class=250):
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.n_class = n_class

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.classes_ = np.array([i for i in range(n_class)])

    def fit(self, X, y):
        self.model.fit(X, y)
        self.model.classes_ = np.array([i for i in range(self.n_class)])

    def predict(self, X):
        probas = self.model.predict_proba(X)
        return probas

    def reset(self):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.classes_ = np.array([i for i in range(self.n_class)])


def knn_eval(X, y, split=0.25, distance="euclidian", n_classes=250):
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

    # fit model no training data
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    model.classes_ = np.array([i for i in range(n_classes)])

    # make predictions for test data
    probas = model.predict_proba(X_test)

    return probas, y_test
