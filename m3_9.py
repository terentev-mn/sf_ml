#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, plot_roc_curve, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import timeit


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

        for _ in range(n_iters):
            theta_grad = self.grad_func(X, y, theta)
            theta = self.gradient_step(theta, theta_grad)

        return theta

    def fit(self, X, y):
        m = X.shape[1]
        start_theta = np.ones(m)
        self.theta = self.optimize(X, y, start_theta, self._n_iters)

    def predict(self, X):
        raise NotImplementedError()


class LogReg(RegOptimizer):
    def sigmoid(self, X, theta):
        return 1. / (1. + np.exp(-X.dot(theta)))

    def grad_func(self, X, y, theta):
        n = X.shape[0]
        grad = 1. / n * X.transpose().dot(self.sigmoid(X, theta) - y)

        return grad

    def predict_proba(self, X):
        return self.sigmoid(X, self.theta)

    def predict(self, X):
        if self.theta is None:
            raise Exception('You should train the model first')

        y_pred = self.predict_proba(X) > 0.5

        return y_pred


def print_logisitc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'acc = {acc:.2f} F1-score = {f1:.2f}')

def print_logisitc_metrics2(y_true, y_pred):
    #acc = accuracy_score(y_true, y_pred)
    return f1_score(y_true, y_pred)


def calc_and_plot_roc(y, y_pred):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)

    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.title('Rereiver Operating Rate (ROC)', fontsize=15)
    plt.xlabel('False positive rate (fpr)', fontsize=15)
    plt.ylabel('True positive rate (tpr)', fontsize=15)
    plt.legend(fontsize=15)
    plt.show()


adult = pd.read_csv('./39_adult.data',
                    names=['age', 'workclass', 'fnlwgt', 'education',
                           'education-num', 'marital-status', 'occupation',
                           'relationship', 'race', 'sex', 'capital-gain',
                           'capital-loss', 'hours-per-week', 'native-country', 'salary'])


'''
Задание 3.9.1
Постройте модель логистической регрессии при помощи sklearn.
Используйте параметры по умолчанию, обучите на всей выборке и посчитайте F1 score.
Ответ округлите до сотых, пример ввода: 0.55
'''
# Избавиться от лишних признаков
adult.drop(['native-country'], axis=1, inplace=True)
# Сконвертировать целевой столбец в бинарные значения
adult['salary'] = (adult['salary'] != ' <=50K').astype('int32')
# Сделать one-hot encoding для некоторых признаков
adult = pd.get_dummies(adult, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex'])
#print(adult.head())

# Нормализовать нуждающиеся в этом признаки
a_features = adult[['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']].values
norm_features = (a_features - a_features.mean(axis=0)) / a_features.std(axis=0)
adult.loc[:, ['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']] = norm_features
#print(adult.head())

# Разбить таблицу данных на матрицы X и y
X = adult[list(set(adult.columns) - set(['salary']))].values
y = adult['salary'].values
# Добавить фиктивный столбец единиц (bias линейной модели)
X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
#m = X.shape[1]

# model
model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)

print_logisitc_metrics(y, y_pred)

'''
Задание 3.9.2
Посчитайте confusion matrix для классификатора из задачи 3.9.1.
Введите значения получившейся матрицы в соответствующие ячейки:
'''
print(confusion_matrix(y, y_pred))

'''
Задание 3.9.3
Постройте ROC-кривую и посчитайте ROC-AUC для классификатора из задачи 3.9.1.
Ответ округлите до сотых, укажите через точку:
'''
#y_pred_proba = model.predict_proba(X)
#print('roc_auc_score', roc_auc_score(y, y_pred_proba[:,1]))
#calc_and_plot_roc(y, y_pred_proba[:,1])

# maybe?
##---plot_roc_curve(model, X, y_pred_proba)
##---plt.show()

'''
Задание 3.9.4
Постройте модель логистической регрессии при помощи sklearn без регуляризации.
Чему равен F1 score?
Ответ округлите до сотых, пример ввода: 0.55
'''
#model = LogisticRegression(penalty='none')
#model.fit(X, y)

#y_pred = model.predict(X)
#print_logisitc_metrics(y, y_pred)


'''
Задание 3.9.5
Переберите коэффициенты l2-регуляризации от 0.01 до 1 с шагом 0.01 и определите,
 на каком из них модель логистической регрессии из sklearn даёт наибольший F1 score.

Ответ округлите до сотых, пример ввода: 0.55
'''
'''
f1 = {}
for i in np.arange(0.01, 1., 0.01):
    model = LogisticRegression(C=i)
    model.fit(X, y)

    y_pred = model.predict(X)
    f1[i] = print_logisitc_metrics2(y, y_pred)

print('i with max f1', max(f1, key=f1.get))
'''
'''
Задание 3.9.6
Замените в столбце native-country страны,
у которых меньше ста записей на other,
поменяйте эту колонку на dummy-переменные,
обучите классификатор на всей выборке и посчитайте F1 score.

Ответ округлите до сотых, пример ввода: 0.55
'''
'''
adult = pd.read_csv('./39_adult.data',
                    names=['age', 'workclass', 'fnlwgt', 'education',
                           'education-num', 'marital-status', 'occupation',
                           'relationship', 'race', 'sex', 'capital-gain',
                           'capital-loss', 'hours-per-week', 'native-country', 'salary'])


# заменяем на other
g = adult.groupby('native-country')['native-country']
gg = g.transform('size')
adult.loc[gg<100, 'native-country'] = 'Other'
#print(adult['native-country'].unique())

# Сконвертировать целевой столбец в бинарные значения
adult['salary'] = (adult['salary'] != ' <=50K').astype('int32')
# Сделать one-hot encoding для некоторых признаков
adult = pd.get_dummies(adult, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
print(adult.head())

# Нормализовать нуждающиеся в этом признаки
a_features = adult[['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']].values
norm_features = (a_features - a_features.mean(axis=0)) / a_features.std(axis=0)
adult.loc[:, ['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']] = norm_features
#print(adult.head())

# Разбить таблицу данных на матрицы X и y
X = adult[list(set(adult.columns) - set(['salary']))].values
y = adult['salary'].values
# Добавить фиктивный столбец единиц (bias линейной модели)
X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
#m = X.shape[1]

# model
model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)

print_logisitc_metrics(y, y_pred)
'''
'''
Задание 3.9.7
Провалидируйте логистическую регрессию из sklearn на 5-fold кросс-валидации.
В логистической регрессии надо выставить random_state=42.
Какой получился средний F1 score ?

Ответ округлите до сотых, пример ввода: 0.55
'''
# http://zabaykin.ru/?p=667
from sklearn.model_selection import cross_val_score
model = LogisticRegression(random_state=42)
model.fit(X, y)

y_pred = model.predict(X)
print_logisitc_metrics(y, y_pred)

# передаем классификатор, X, y и кол-во фолдов=5
res = cross_val_score(model, X, y, cv=5, scoring='f1')
print(res)
print(np.mean(res))