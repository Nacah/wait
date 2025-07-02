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


    df[col].unique() #-> return a list of unique values in the series
df[col].nunique() #-> return the number of unique values in the series
df.isnull().sum() #-> return the number of null values in the dataframe

    
plt.figure(figsize=(6, 4)) #**размер графика

sns.histplot(df.msrp, bins=40, color='black', alpha=1) #** Хистограмма распределения цен
plt.ylabel('Count') #** Кол во машин
plt.xlabel('Price') #** Цены
plt.title('Distribution of prices') #** Название графика


plt.figure(figsize=(6, 4))

sns.histplot(df.msrp[df.msrp < 100000], bins=40, color='black', alpha=1)  #** Хистограмма распределения цен которые ниже 100 тысяч
plt.ylabel('Frequency')
plt.xlabel('Price')
plt.title('Distribution of prices')
plt.show()



log_price = np.log1p(df.msrp)

plt.figure(figsize=(6, 4))

sns.histplot(log_price, bins=40, color='black', alpha=1)  #** Так как разброс в ценах слищком большой и разный нужно взять логарифмы для более точного анализа
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


///ЛИНЕЙНАЯ РЕГРЕССИЯ///

Сначала для представления что такое лин. регрессия попробуем сделать её только на основе одной машины
g(xi)=~yi  xi:машина  yi:предположение цены g:модель
берем 3 признака(фичи) = engine_hp,city_mpg,popularity
xi = [453,11,86]
g(xi) = w0 + w1*xi1 + w2*xi2 + w3*xi3 # w0: свободный коэффицент; w1,w2,w3: веса 1,2,3 признака; xi1,xi2,xi3: 1,2,3 признаки
w0 = 7.17
w = [0.01, 0.04, 0.002]

#def linear_regression(xi):
    #n = len(xi)

    #pred = w0

    #for j in range(n):
        #pred = pred + w[j] * xi[j]
   # return pred ===> 12.312

##Версия функции с dot()   
w_new = [w0] + w  
def dot(xi,w):
    n = len(xi)

    res = 0.0

    for j in range(n):
        res = res + w[j] * xi[j]
    return res
   
def linear_regression(xi):
    xi = [1] + xi
    return dot(xi, w_new)   
#После получения pred или res это не окончательная цена так как мы взяли логарифмы цен в коде выше
поэтому нужно взять экспонент pred
#np.expm1(12.312) ===> 222347.2221101062

###Векторная Форма Линейной Регрессии###
Теперь нашу функцию нужно привести из простой формы в более сложную
скажем мы возьмем 3 фичи 3 машин:
w0 = 7.17
w = [0.01, 0.04, 0.002]
w_new = [w0] + w ===> [7,17, 0.01, 0.04, 0.002]
x1 = [1, 148, 24, 1385]
x2 = [1, 132, 25, 2031] 
x10 = [1, 453, 11, 86]
# Как можно заметить появился стобец с единицами. Столбец с единицами нужен, чтобы модель могла вычислить свободный коэффициент (w₀, или bias / intercept).
# Обычная линейная модель без столбца единиц: y = w1x1 + w2x2 + ... + wnxn **Проблема: если все фичи равны 0 
а в реальности — чаще всего есть какое-то начальное значение даже при нулевых признаках. То есть w0 это значение(вес) которое обозначает наше начальное предположение до просмотра фич
# поэтому нужны единицы дабы обозначить начальный вес

X = [x1, x2, x10]
X = np.array(X)

def linear_regression(X):
    return X.dot(w_new)

///Обучение Линейной Регрессии: Нормальное уравнение///
** Как было сказано до этого мы тренируем модель g(X) = y чтобы найти оптимальную цену, для этого нужно найти веса (w) так как g(X) = wX уравнение будет таким:
wX = y и его нужно решить относительно w чтобы найти веса нашей лин. регрессии
Для уравнения используют термин "Нормальное уравнение(Normal Equation)"
w = X-1 * y не реально потому что X это в 99% процентов случаев не квадратная матриуа поэтому не можем инверировать ее и поэтому уравнение невозможно решить с точным ответом
поэтому нужно найти примерное решение данного уравнение для этого и нужно нормальное уравнение
Решение:
XT * X * w = XT * y  ! С двух сторон умножаем транспонированным X  ##Примечание: у "XT * X" есть специальное название "GRAM MATRIX" 
И уже GRAM MATRIX мы можем инвертировать
(XT*X)**-1 * w = XT * y
w = (XT*X)**-1 * XT * y ===> ###ЭТО УРАВНЕНИЕ ДАЕТ НАМ САМОЕ БЛИЗКОЕ РЕШЕНИЕ СИСТЕМЫ 


#** NORMAL EQUATION В ПАЙТОНЕ 

X = [
[148, 24, 1385],
[132, 25, 2031],
[453, 11, 86],
[158, 24, 185],
[172, 25, 201],
[413, 11, 86],
[38, 54, 185],
[142, 25, 431],
[453, 31, 86],
]
X = np.array(X)
y = [100, 200, 150, 250, 100, 200, 150, 250, 120]
def train_linear_regression(X, y):

    ones = np.ones(X.shape[0])  # Создание столбца нулевых признаков из единиц

    X = np.column_stack([ones, X])  # Добавление столбца нулевых признаков из единиц

    XTX = X.T.dot(X) # Создание gram matrix транспонированная матрица умноженная на оригинальную матрицу

    XTX_inv = np.linalg.inv(XTX) # Инвертирование gram matrix

    w = XTX_inv.dot(X.T).dot(y) # Вычисление весов
    
    return w[0], w[1:] # Нулевой вес и весы признаков
print(train_linear_regression(X,y))


np.random.seed(2) #** Закрепоение рандомно разбросанного датафрейма

n = len(df) #** Кол.во записей

n_val = int(0.2 * n)  #** 20%
n_test = int(0.2 * n)  #** 20%
n_train = n - (n_val + n_test)  #** 60%

idx = np.arange(n)  #** Индексы записей
np.random.shuffle(idx)  #** Рандомный разброс индексов
#idx ===> чтобы проверить раднмно или нет выведет рандомные индексы ===> [10200,1400,23, ....,11001]
df_shuffled = df.iloc[idx]  

df_train = df_shuffled.iloc[:n_train].copy()   #** Часть для обучения модели train
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy() #** Часть для валидации
df_test = df_shuffled.iloc[n_train+n_val:].copy()  #** Часть для теста
y_train_orig = df_train.msrp.values  
y_val_orig = df_val.msrp.values
y_test_orig = df_test.msrp.values

y_train = np.log1p(df_train.msrp.values)  #** predictions или y
y_val = np.log1p(df_val.msrp.values)  #** predictions или y
y_test = np.log1p(df_test.msrp.values)  #** predictions или y

del df_train['msrp']
del df_val['msrp']  #*** Удаление Созданных стобцов из датафрейма чтобы случайно не использовать их
del df_test['msrp']

#///Создание baseline модели
def train_linear_regression(X, y):

    ones = np.ones(X.shape[0])  # Создание столбца нулевых признаков из единиц

    X = np.column_stack([ones, X])  # Добавление столбца нулевых признаков из единиц

    XTX = X.T.dot(X) # Создание gram matrix транспонированная матрица умноженная на оригинальную матрицу

    XTX_inv = np.linalg.inv(XTX) # Инвертирование gram matrix

    w = XTX_inv.dot(X.T).dot(y) # Вычисление весов
    
    return w[0], w[1:]

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

def prepare_X(df):    # Функция отвечающая за подготовку матрицы признаков в частности за заполнение пустых значений нулями
    df_num = df[base]  
    df_num = df_num.fillna(0) # Заполнение пустых значений нулями
    X = df_num.values
    return X
X_train = prepare_X(df_train)  # Заполнение пустых значений
w_0, w = train_linear_regression(X_train, y_train) # Использование моделиб вычисление весов
y_pred = w_0 + X_train.dot(w)  # Предположение суммы
plt.figure(figsize=(6, 4))

print(sns.histplot(y_train, label='target', color='#222222', alpha=0.6, bins=40))       # Построение двух хистограмм одна со значениями из датасета, вторая со згачениями которые мы предположили
print(sns.histplot(y_pred, label='prediction', color='#aaaaaa', alpha=0.8, bins=40))                               

plt.legend()

plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Predictions vs actual distribution')

///Root Mean Squared Error///

у нас есть array с реальными значениями цен и с предсказанными ценами, нужно найти ошибку между ними
Мы от предсказанных значений отнимаем реальные значения, возводим в квадрат, суммируем все значения и делим на кол-во значений, после чего берем корень из полученного числа
def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
rmse(y_val, y_pred)

### Featue engineering ###
#** Feature engineering это процесс создания новых признаков из имеющихся, чтобы улучшить качество модели
#** Например, можно создать новый признак, который будет представлять собой возраст машины, вычитая год выпуска из текущего года

def prepare_X(df):
    df = df.copy() ===> # Создаем копию датафрейма, чтобы не изменять оригинал
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)

y_pred = w_0 + X_train.dot(w)
print('train', rmse(y_train, y_pred))

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('validation', rmse(y_val, y_pred))
train 0.5175055465840046
validation 0.5172055461058335
plt.figure(figsize=(6, 4))


sns.histplot(y_val, label='target', color='#222222', alpha=0.6, bins=40)
sns.histplot(y_pred, label='prediction', color='#aaaaaa', alpha=0.8, bins=40)

plt.legend()

plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Predictions vs actual distribution')

plt.show()
"""




 