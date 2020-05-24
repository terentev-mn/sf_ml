#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Задание 2.9.3
# Загрузите данные train.csv, найдите признак, обозначающий баланс.
# Для приведения данных к более нормальному виду часто используют
# различные преобразования, например, взятие корня от признака.
# Возьмите корень у положительных значений, найдите медианное и среднее значение.
# В качестве ответа укажите модуль разницы этих значений.

vis_data = pd.read_csv("train.csv", encoding = 'ISO-8859-1', low_memory = False)
# print(vis_data.info()) # находим столбец 'balance_due'
# берем столбец 'balance_due' (получаем массив)
balance = vis_data.balance_due.values
'''
# размер массива
balance_size = balance.size
# преобразуем массив в матрицу
bal_matrix = balance.reshape(balance_size, 1)

scaler = StandardScaler()
balance_norm = scaler.fit_transform(bal_matrix)
balnce_norm_min = balance_norm.min()
print('balnce_norm_min', round(balnce_norm_min, 5))
'''