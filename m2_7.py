#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Задание 2.7.3
# Загрузите данные train.csv, найдите признак, обозначающий баланс.
# Нормализуйте признак с помощью std-нормализации.
# Проверьте, что вы нашли нужный признак и нормализовали его подходящим методом.
# Метод для нормализации принимает матрицу, а не массив.
# В numpy можно превратить массив в матрицу с помощью reshape().
# В качестве ответа укажите минимальное значение в получившемся нормализованном признаке.
# Ответ округлите до 5 знаков после запятой.

vis_data = pd.read_csv("train.csv", encoding = 'ISO-8859-1', low_memory = False)
# print(vis_data.info()) # находим столбец 'balance_due'
# берем столбец 'balance_due' (получаем массив)
balance = vis_data.balance_due.values
# размер массива
balance_size = balance.size
# преобразуем массив в матрицу
bal_matrix = balance.reshape(balance_size, 1)

scaler = StandardScaler()
balance_norm = scaler.fit_transform(bal_matrix)
balnce_norm_min = balance_norm.min()
print('balnce_norm_min', round(balnce_norm_min, 5))
'''
# исследуем столбец 'state'
# вариант №1
print(vis_data.state.value_counts(dropna=False))
# вариант №1
mode_state = vis_data.state.mode()
print('значение, которое встречается чаще всего в "state"', mode_state)
# тут надо извлечь из dtype часть 'MI'
result = vis_data.state.fillna('MI')
'''