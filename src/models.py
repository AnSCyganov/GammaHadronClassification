from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def knn_model(X_train, y_train, X_test, y_test, n_neighbors):
    if n_neighbors == 0:
        return "Количество соседей не может быть меньше 1"
    else:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report