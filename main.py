from src.data import get_magic_col_names, load_data, set_binary_class, count_classes, split_data, scale_transform, scale_fit_transform

#Работа с данными
col_names = get_magic_col_names()
DataFrame = load_data('magic04.data', col_names)
print(DataFrame.head())
set_binary_class(DataFrame, 'g')
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

X_train, y_train, scaler = scale_fit_transform(train, scaler, oversample=True)
X_valid, y_valid, scaler = scale_transform(valid, scaler)
X_test, y_test, scaler = scale_transform(test, scaler)

from src.models import train_and_evaluate

param_grid = [
{
    'C': [0.1, 1, 10],
    'kernel': ['linear'],
    'class_weight': [None, 'balanced']
},
{
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale', 0.01, 0.1, 1],
    'class_weight': [None, 'balanced']
}
]

train_and_evaluate(model_name='svm',
                   X_train=X_train, y_train=y_train,
                   X_test=X_test, y_test=y_test,
                   param_grid=param_grid,
                   scoring='f1',
                   cv=5)