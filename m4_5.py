#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

n_samples = 1500
dataset = datasets.make_blobs(n_samples=n_samples, centers=2, center_box=(-7.0, 7.5),
                              cluster_std=[1.4, 1.7],
                              random_state=42)
X_2, _ = datasets.make_blobs(n_samples=n_samples, random_state=170, centers=[[-4, -3]], cluster_std=[1.9])
transformation = [[1.2, -0.8], [-0.4, 1.7]]
X_2 = np.dot(X_2, transformation)
X, y = np.concatenate((dataset[0], X_2)), np.concatenate((dataset[1], np.array([2] * len(X_2))))

#Визуализируем наш датасет:
#plt.rcParams['figure.figsize'] = 10, 10
#plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
#plt.show()

#Посмотрим распределение классов в датасете:
#unique, counts = np.unique(y, return_counts=True)
#print(dict(zip(unique, counts)))

'''
Задание 4.5.2
Обучите модель K-means с параметрами n_clusters=3 и random_state=42 на признаках исходного датасета.
Какие центроиды будут у получившихся кластеров?
Введите ответ в виде массива. Каждое число в ответе округлите до ближайшего целого.
'''
k_means = KMeans(n_clusters=3, random_state=42)
k_means.fit(X)
a = k_means.cluster_centers_
print(np.round(a).astype(np.int))
#print(k_means.cluster_centers_)


'''
Задание 4.5.3
Подсчитайте количество элементов в каждом из получившихся кластеров.
Запишите в форму ниже три числа через пробел(без запятых!):
 количество элементов в кластере 0, в кластере 1 и в кластере 2.
 Записывайте строго в таком порядке.

Для подсчёта элементов в списке можно воспользоваться функцией numpy.unique с параметром return_counts=True:
'''
from collections import Counter, defaultdict
print(Counter(k_means.labels_))







