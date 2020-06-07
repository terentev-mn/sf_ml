#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns # библиотека для визуализации статистических данных
import matplotlib.pyplot as plt # для построения графиков


# Задание 2.9.3
# Загрузите данные train.csv, найдите признак, обозначающий баланс.
# Для приведения данных к более нормальному виду часто используют
# различные преобразования, например, взятие корня от признака.
# Возьмите корень у положительных значений, найдите медианное и среднее значение.
# В качестве ответа укажите модуль разницы этих значений.

vis_data = pd.read_csv("train.csv", encoding = 'ISO-8859-1', low_memory = False)
# print(vis_data.info()) # находим столбец 'balance_due'
# берем столбец 'balance_due' (получаем массив)
#balance = vis_data.balance_due.values
#plt.figure(figsize=(16,6))
a = np.sqrt(vis_data.balance_due[vis_data.balance_due > 0])
print('median', np.median(a))
print('mean', np.mean(a))
print('ABS', np.abs(np.median(a) - np.mean(a)))
#a = np.log(vis_data.balance_due[vis_data.balance_due > 0]).hist()
#print(a)
#plt.show()


#b = np.abs(a)
#print(b)
'''
plt.figure(figsize=(16,6))
heat_map1 = sns.heatmap(data=re1_corr, annot=True)
#heat_map0 = sns.heatmap(data=re0_corr, annot=True)
plt.show()

# размер массива
balance_size = balance.size
# преобразуем массив в матрицу
bal_matrix = balance.reshape(balance_size, 1)

scaler = StandardScaler()
balance_norm = scaler.fit_transform(bal_matrix)
balnce_norm_min = balance_norm.min()
print('balnce_norm_min', round(balnce_norm_min, 5))
'''
