import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

def load_data(name, col_names):
    data = pd.read_csv('data/' + name, names=col_names)
    return data

def set_col_names(name):
    if type(name) is not list:
        print(f"Предупреждение: требуется тип данных : {list}")
    else:
        return name

def set_binary_class(df):
    if len(df['class'].unique()) != 2:
        raise ValueError(f'Количество классов должно быть 2 !')
    else:
        df['class'] = (df['class'] == 'g').astype(int)

def count_classes(df):
    return len(df[df['class'] == 1]), len(df[df['class'] == 0])

def split_data(df):
    train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])
    return train, valid, test

def scale_fit_transform(train_df, scaler, oversample=False):
    X = train_df[train_df.columns[:-1].values]
    y = train_df['class'].values

    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X,y = ros.fit_resample(X,y)

    return X, y

def scale_transform(df, scaler):
    X = df[df.columns[:-1].values]
    y = df['class'].values

    X = scaler.transform(X)

    return X, y