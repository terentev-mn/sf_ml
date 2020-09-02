#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures



vis_data = pd.read_csv("train.csv", encoding = 'ISO-8859-1', low_memory = False)
#print(vis_data.info())
'''
#Задание 2.11.6
#Загрузите данные train.csv, найдите признак, обозначающий баланс и признак,
#обозначающий размер скидки. Создайте полиномиальные признаки степени 3.
#Посчитайте среднее значение для каждого получившегося признака.
#В качестве ответа укажите номер признака, который содержит максимальное среднее значение.

pf = PolynomialFeatures(3)
poly_features = pf.fit_transform(vis_data[['balance_due', 'discount_amount']])
m = poly_features.mean(axis=0)
#print(poly_features)
#print(poly_features.shape)
#print(max(m))
print(np.argmax(m, axis=0))
'''

#Задание 2.11.7
#Загрузите данные train.csv, найдите признак, обозначающий дату, когда был выписан штраф.
#Найдите, сколько раз штраф был выписан на выходных и запишите это число в качестве ответа.
#Выходными считаются дни под номерами 5 и 6.

# ticket_issued_date
datetime_vals = pd.to_datetime(vis_data.ticket_issued_date.dropna())
#print(datetime_vals.head())
vis_data['is_weekend'] = datetime_vals.dt.weekday > 4
print(vis_data['is_weekend'].value_counts()) # 1620