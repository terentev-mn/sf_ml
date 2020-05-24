#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Задание 2.5.6
# Загрузите данные train.csv, найдите признак, обозначающий штат.
# Затем найдите значение, которое встречается чаще всего.
# Замените пропуски этим значением и запишите получившийся признак в переменную result.
vis_data = pd.read_csv("train.csv", encoding = 'ISO-8859-1', low_memory = False)
# исследуем столбец 'state'
# вариант №1
print(vis_data.state.value_counts(dropna=False))
# вариант №1
mode_state = vis_data.state.mode()
print('значение, которое встречается чаще всего в "state"', mode_state)
# тут надо извлечь из dtype часть 'MI'
result = vis_data.state.fillna('MI')
