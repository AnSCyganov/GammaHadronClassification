import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

def load_data(name, col_names):
    data = pd.read_csv('data/' + name, names=col_names)
    return data

def get_magic_col_names():
    return ['fLenght', 'fWidth', 'fSize', 'fConc', 'fConcl', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

def set_binary_class(df, positive_class):
    unique_classes = df['class'].unique()
    if len(unique_classes) != 2:
        raise ValueError(f'Ожидалось 2 класса, найдено {unique_classes}')
    df['class'] = (df['class'] == positive_class).astype(int)

def count_classes(df):
    return len(df[df['class'] == 1]), len(df[df['class'] == 0])

def split_data(df):
    shuffled_df = df.sample(frac=1, random_state = 42).reset_index(drop=True)

    train_size, valid_size = int(0.6 * len(shuffled_df)), int(0.8 * len(shuffled_df))
    train, valid, test = np.split(shuffled_df, [train_size, valid_size])

    return train, valid, test

def scale_fit_transform(train_df, scaler, oversample=False):
    X = train_df[train_df.columns[:-1].values]
    y = train_df['class'].values

    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X,y = ros.fit_resample(X,y)

    return X, y, scaler

def scale_transform(df, scaler):
    X = df[df.columns[:-1].values]
    y = df['class'].values

    X = scaler.transform(X)

    return X, y, scaler