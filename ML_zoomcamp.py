import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

"""
df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')  #Работа с названиями колонок, уменшили буквы и поставили нижний пробел для читабельности
strings = list(df.dtypes[df.dtypes == "object"].index)  # обозначили какие колонки меют строковые значения
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')  #Работа с элементами колонок, уменьшили буквы и поставили нижний пробел для читабельности


    df[col].unique() -> return a list of unique values in the series
df[col].nunique() -> return the number of unique values in the series
df.isnull().sum() -> return the number of null values in the dataframe

    
plt.figure(figsize=(6, 4)) **размер графика

sns.histplot(df.msrp, bins=40, color='black', alpha=1) ** Хистограмма распределения цен
plt.ylabel('Count') ** Кол во машин
plt.xlabel('Price') ** Цены
plt.title('Distribution of prices') ** Название графика


plt.figure(figsize=(6, 4))

sns.histplot(df.msrp[df.msrp < 100000], bins=40, color='black', alpha=1)  ** Хистограмма распределения цен которые ниже 100 тысяч
plt.ylabel('Frequency')
plt.xlabel('Price')
plt.title('Distribution of prices')
plt.show()



log_price = np.log1p(df.msrp)

plt.figure(figsize=(6, 4))

sns.histplot(log_price, bins=40, color='black', alpha=1)  ** Так как разброс в ценах слищком большой и разный нужно взять логарифмы для более точного анализа
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Distribution of prices after log tranformation')

plt.show()


///Пустые места///
df.isnull().sum()
make                    0
model                   0
year                    0
engine_fuel_type        3
engine_hp              69
engine_cylinders       30
transmission_type       0
driven_wheels           0
number_of_doors         6
market_category      3742
vehicle_size            0
vehicle_style           0
highway_mpg             0
city_mpg                0
popularity              0
msrp                    0
dtype: int64


///Validation///

**Для построения модели как было написано в заметках, Данные делятся на 3 части: train это 60% всех данных, validation 20%, test 20% 
перед делением на части:
1. Нужно найти 60%, 20% и 20% процентов данных в числовом формате т.е. сколько это записей будет
2. Данные нужно рандомно разбросать для большей точности потому что данные построены по порядку марки автомобиля

np.random.seed(2) ** Закрепоение рандомно разбросанного датафрейма

n = len(df) ** Кол.во записей

n_val = int(0.2 * n)  ** 20%
n_test = int(0.2 * n)  ** 20%
n_train = n - (n_val + n_test)  ** 60%

idx = np.arange(n)  ** Индексы записей
np.random.shuffle(idx)  ** Рандомный разброс индексов
idx ===> чтобы проверить раднмно или нет выведет рандомные индексы ===> [10200,1400,23, ....,11001]
df_shuffled = df.iloc[idx]  

df_train = df_shuffled.iloc[:n_train].copy()   ** Часть для обучения модели train
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy() ** Часть для валидации
df_test = df_shuffled.iloc[n_train+n_val:].copy()  ** Часть для теста
y_train_orig = df_train.msrp.values  
y_val_orig = df_val.msrp.values
y_test_orig = df_test.msrp.values

y_train = np.log1p(df_train.msrp.values)  ** predictions или y
y_val = np.log1p(df_val.msrp.values)  ** predictions или y
y_test = np.log1p(df_test.msrp.values)  ** predictions или y

del df_train['msrp']
del df_val['msrp']  *** Удаление Созданных стобцов из датафрейма чтобы случайно не использовать их
del df_test['msrp']

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:] 

"""    