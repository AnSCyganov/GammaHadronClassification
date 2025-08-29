from src.data import set_col_names, load_data, set_binary_class, count_classes, split_data, scale_transform, scale_fit_transform

#Работа с данными
col_names = set_col_names(['fLenght', 'fWidth', 'fSize', 'fConc', 'fConcl', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class'])
print(col_names)
DataFrame = load_data('magic04.data', col_names)
print(DataFrame.head())
set_binary_class(DataFrame)
print(f"\n{DataFrame.head()}")
print(f"\n{list(count_classes(DataFrame))}")

#Разделение на тренировочные, валидационные и тестовые данные
train, valid, test = split_data(DataFrame)
print(f"Count gamma and hadron in train: {list(count_classes(train))}")
print(f"Count gamma and hadron in valid: {list(count_classes(valid))}")
print(f"Count gamma and hadron in test: {list(count_classes(test))}")

from src.visualize import visualize

#Визуализация
#visualize(DataFrame)

#Нормализация данных
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train, y_train = scale_fit_transform(train, scaler, oversample=True)
X_valid, y_valid = scale_transform(valid, scaler)
X_test, y_test = scale_transform(test, scaler)

from src.models import knn_model
for k in [5, 7, 9, 11, 13, 15, 17]:
    print(f"Count of neighbors : {k} \n report : \n{knn_model(X_train, y_train, X_test, y_test, k)}")

from src.models import svm_model

C_values = [0.1, 1, 10]
kernels = ["linear", "rbf"]
gammas = ["scale", 0.01, 0.1, 1]
class_weights = [None, "balanced"]

for C in C_values:
    for kernel in kernels:
        for class_weight in class_weights:
            if kernel == "linear":
                print(f"\nC={C}, kernel={kernel}, class_weight={class_weight}\n"
                      f"{svm_model(X_train, y_train, X_test, y_test, C=C, kernel=kernel, class_weight=class_weight)}")
            else:
                for gamma in gammas:
                    print(f"\nC={C}, kernel={kernel}, class_weight={class_weight}, gamma={gamma}\n"
                          f"{svm_model(X_train, y_train, X_test, y_test, C=C, kernel=kernel, class_weight=class_weight, gamma=gamma)}")