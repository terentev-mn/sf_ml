#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

# загрузим датасет Boston Housing
data = load_boston()
X, y = data['data'], data['target']
# добавить единички
X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

def linreg_linear(X, y):
    #theta = (X.T@X)**(-1)*Xt*y
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta


def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE = {mse:.2f}, RMSE = {rmse:.2f}')


'''
Задание 3.5.1
Реализуйте матричную линейную регрессию. Какой получился RMSE?
Ответ округлите до сотых, пример ввода: 0.55
'''
#theta = linreg_linear(X, y)
#y_pred = X @ theta
#print_regression_metrics(y, y_pred)

'''
Задание 3.5.2
Постройте модель при помощи sklearn.
Используйте параметры по умолчанию, обучите на всей выборке и посчитайте RMSE.
Ответ округлите до сотых, пример ввода: 0.55
'''
#from sklearn import linear_model
#from sklearn.linear_model import LinearRegression
# обучаем модель
#lm1 = linear_model.LinearRegression()
#model1 = lm1.fit(X, y)

#y_pred = model1.predict(X)
#print_regression_metrics(y, y_pred)


'''
Задание 3.5.3
У какого из признаков наибольшее стандартное отклонение? Чему оно равно?
При подсчёте, в функции std нужно указать параметр ddof=0 (или 1, если там 0).
Ответ округлите до сотых, пример ввода: 155.55
'''
#for i in range(X.shape[1]):
#    XX = X[:,i].std(axis=0, ddof=0)
#    print(XX.max())


'''
Задание 3.5.4
Обучите регрессию без дополнительного столбца единиц. Какой получился RMSE?
Ответ округлите до сотых, пример ввода: 5.55
'''
#X, y = data['data'], data['target']
#theta = linreg_linear(X, y)
#y_pred = X @ theta
#print_regression_metrics(y, y_pred)

'''
Задание 3.5.5
Очистите данные от строк, где значение признака B меньше 50. Какой получился RMSE?
Ответ округлите до сотых, пример ввода: 5.55
'''
#X = data.data[data.data[:,11] > 50]
#y = data.target[data.data[:,11] > 50]
#X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

#theta = linreg_linear(X, y)
#y_pred = X @ theta
#print_regression_metrics(y, y_pred)

'''
Задание 3.5.6
Нормализуйте признаки и обучите линейную регрессию матричным методом. Какой получился RMSE?
Ответ округлите до сотых, пример ввода: 5.55
'''
X, y = data['data'], data['target']
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_norm = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

theta = linreg_linear(X_norm, y)
y_pred = X_norm @ theta
print_regression_metrics(y, y_pred)







