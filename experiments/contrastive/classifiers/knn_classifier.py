from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_eval(X, y, split=0.25, distance="euclidian", n_labels=10):
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

    # fit model no training data
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # make predictions for test data
    probas = model.predict_proba(X_test)

    return probas, y_test
