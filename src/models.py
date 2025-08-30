from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

MODELS = {
    'knn' : KNeighborsClassifier,
    'svm' : SVC
}

def create_model(model_name, **kwargs):
    if model_name not in MODELS:
        raise ValueError(f'Модель {model_name} не найдена'
                         f'\nДоступные модели: {list(MODELS.keys())}')
    return MODELS[model_name](**kwargs)

def grid_search(model_name, param_grid, X_train, y_train, scoring=None, cv=5, n_jobs=-1):
    model = create_model(model_name=model_name)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

def train_and_evaluate(model_name, X_train, y_train, X_test, y_test, param_grid=None, scoring=None, cv=5, **kwargs):
    if param_grid:
        base_model = create_model(model_name)
        model, best_params, best_score = grid_search(model_name=model_name, param_grid=param_grid, X_train=X_train, y_train=y_train, scoring=scoring, cv=cv)
        print(f'Топ модель: {model} с параметрами: {best_params} и скором: {best_score}')
    else:
        model = create_model(model_name, **kwargs)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f'Отчет о модели: {model_name}:\n{classification_report(y_test, y_pred)}')

    return model