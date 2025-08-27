from src.data import set_col_names, load_data, set_binary_class, count_classes, split_data, Scale_DF

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
train, X_train, y_train = Scale_DF(train, oversample=True)
valid, X_valid, y_valid = Scale_DF(valid, oversample=False)
test, X_test, y_test = Scale_DF(test, oversample=False)

from src.models import knn_model
for k in [1, 2, 3, 5, 7, 9, 11]:
    print(knn_model(X_train, y_train, X_test, y_test, k))