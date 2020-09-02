#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns # библиотека для визуализации статистических данных
import matplotlib.pyplot as plt # для построения графиков


# Задание 2.13.5
# Загрузите данные train.csv, найдите признак, обозначающий баланс.
# Уберите пропуски из этого признака и найдите выбросы с помощью межквартильного расстояния.
# Найдите модуль разницы между минимальным и максимальным выбросом.

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))[0]


vis_data = pd.read_csv("train.csv", encoding = 'ISO-8859-1', low_memory = False)
#print(vis_data.info()) # находим столбец 'balance_due'
b = outliers_iqr(vis_data['balance_due']) # возвращает индексы
print(b)
print(vis_data['balance_due'][b].max() - vis_data['balance_due'][b].min())
