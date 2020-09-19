#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import timeit


def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE = {mse:.2f}, RMSE = {rmse:.2f}')

def prepare_boston_data():
    data = load_boston()
    X, y = data['data'], data['target']
    # Нормализовать даннные с помощью стандартной нормализации
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Добавить фиктивный столбец единиц (bias линейной модели)
    X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

    return X, y

class LinRegAlgebra():
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        #self.theta = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X.dot(self.theta)


class RegOptimizer():
    def __init__(self, alpha, n_iters):
        self.theta = None
        self._alpha = alpha
        self._n_iters = n_iters

    def gradient_step(self, theta, theta_grad):
        return theta - self._alpha * theta_grad

    def grad_func(self, X, y, theta):
        raise NotImplementedError()

    def optimize(self, X, y, start_theta, n_iters):
        theta = start_theta.copy()

        for i in range(n_iters):
            theta_grad = self.grad_func(X, y, theta)
            # 3.6.1
            #if theta_grad.max() < 0.01:
            #    print(f'iter {i}')
            #    break
            theta = self.gradient_step(theta, theta_grad)

        return theta

    def fit(self, X, y):
        m = X.shape[1]
        start_theta = np.ones(m)
        self.theta = self.optimize(X, y, start_theta, self._n_iters)

    def predict(self, X):
        raise NotImplementedError()

class LinReg(RegOptimizer):
    # Реализуем функцию grad_func(), которая возвращает градиент, и predict()
    def grad_func(self, X, y, theta):
        n = X.shape[0]
        grad = 1. / n * X.transpose().dot(X.dot(theta) - y)

        return grad

    def predict(self, X):
        if self.theta is None:
            raise Exception('You should train the model first')

        y_pred = X.dot(self.theta)

        return y_pred


def prepare_boston_data_new():
    data = load_boston()
    X, y = data['data'], data['target']

    X = np.hstack([X, np.sqrt(X[:, 5:6]), X[:, 6:7] ** 3])
    # 3.6.2
    #X = np.hstack([X,  X[:, 7:8]** 2])
    # Нормализовать даннные с помощью стандартной нормализации
    # 3.6.3
    #X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Добавить фиктивный столбец единиц (bias линейной модели)
    X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

    return X, y


def train_validate(X, y):
    # Разбить данные на train/valid
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # Создать и обучить линейную регрессию
    linreg_alg = LinRegAlgebra()
    linreg_alg.fit(X_train, y_train)

    # Сделать предсказания по валидционной выборке
    y_pred = linreg_alg.predict(X_valid)

    # Посчитать значение ошибок MSE и RMSE для валидационных данных
    print_regression_metrics(y_valid, y_pred)

'''
Задание 3.6.1
Сделайте для градиентного спуска остановку алгоритма,
 если максимальное из абсолютных значений компонент градиента становится меньше 0.01.
Сравните скорость обучения градиентным спуском и матричными операциями.
На какой итерации останавливается градиентный спуск?
'''
'''
X, y = prepare_boston_data()

start_time = timeit.default_timer()

linreg_crit = LinReg(0.2, 1000)
linreg_crit.fit(X, y)
y_pred = linreg_crit.predict(X)
print_regression_metrics(y, y_pred)
print(timeit.default_timer() - start_time)

#LinRegAlgebra
linreg_crit = LinRegAlgebra()
linreg_crit.fit(X, y)
y_pred = linreg_crit.predict(X)
print_regression_metrics(y, y_pred)
print(timeit.default_timer() - start_time)
'''

'''
Задание 3.6.2
Добавьте к признакам нелинейной модели квадрат признака DIS и переобучите модель.
 Какой получился RMSE?
Ответ округлите до сотых, пример ввода: 5.55
'''
X, y = prepare_boston_data_new()
train_validate(X, y)
#linreg_crit = LinReg(0.2, 1000)
#linreg_crit.fit(X, y)
#y_pred = linreg_crit.predict(X)
#print_regression_metrics(y, y_pred)



