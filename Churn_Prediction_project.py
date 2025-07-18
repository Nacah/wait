"""
ПРЕДСКАЗЫВАНИЕ CHURN (ОТТОКА КЛИЕНТОВ)
Churn — это ситуация, когда пользователи перестают пользоваться продуктом или услугой.
Примеры:
Клиенты перестают платить за подписку (Netflix, Spotify, SaaS).
Пользователи удаляют приложение.
Абоненты отключают мобильную связь.
Датасет используемый в проекте относиться к телеком компании который имеет информацию о клиентах
Модель обучается на задаче прогнозирования churn (оттока клиентов).
Используется sklearn.linear_model.LogisticRegression на закодированных признаках (one-hot).
Анализируются метрики: accuracy, ROC-AUC, матрица ошибок.
Коэффициенты модели интерпретируются через логарифм шансов и/или odds ratio.


"""


import kagglehub
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
df = pd.read_csv(os.path.join(path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
#print(df.head().T) # Транспонировали для более удобного обзора


df.columns = df.columns.str.lower().str.replace(' ', '_') # Работа с названиями колонок, уменшили регистры и поставили нижний пробел для читабельности

object_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in object_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_') # Работа с элементами колонок, уменьшили буквы и поставили нижний пробел для читабельности

df.churn = (df.churn == 'yes').astype(int) # Конвертация yes/no в 0 и 1
df.head().T
strings = list(df.dtypes[df.dtypes==object].index) # обозначили какие колонки имеют строковые значения 
# Что делает: df.dtypes[...].index «Возьми все колонки, где True» .index — верни названия этих колонок
# Если ты хочешь вернуть названия колонок, где условие False, т.е. все остальные типы (не object), просто добавь знак отрицания ~. list(df.dtypes[~(df.dtypes == object)].index)

df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce') # Конвертируем колонку платежей клиентов к численным данным #  errors='coerce' игнорирует значение где мы добавили нижний пробел
df['totalcharges'] = df['totalcharges'].fillna(0)  # Заполняем те места которые проигнорировали ранее, нулями

#Scikit-learn 

# from sklearn.model_selection import train_test_split импортировали в начале
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values
del df_train['churn']
del df_val['churn']

#EDA
df_train_full.isnull().sum() # checking null values
df_train_full.churn.value_counts() #проверка ушедших и оставшихся клиентов
global_mean = df_train_full.churn.mean() # ===> 0.26996805111821087 => CHURN RATE  # средний показатель оттока то есть процентно сколько клиентов ушло(так как там только нули и единицы средняя считается как сумма "1"ок поделенная на кол.во клиентов)
round(global_mean, 3) # ===> 0.27

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents', # КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']

numerical = ['tenure', 'monthlycharges', 'totalcharges'] # ЧИСЛОВЫЕ ПРИЗНАКИ
df_train_full[categorical].nunique() # КОЛ.ВО УНИКАЛЬНЫХ ТИПОВ



# Feature importance (Важность признаков)

# Feature importance — это оценка, какие признаки (фичи) сильнее всего влияют на предсказание модели.
# Она показывает, какие данные важнее для принятия решений моделью, а какие почти не влияют на результат.
"""
female_mean = df_train_full[df_train_full.gender == 'female'].churn.mean() процент оттока женщин
print('gender == female:', round(female_mean, 3))

male_mean = df_train_full[df_train_full.gender == 'male'].churn.mean() процент оттока мужчин
print('gender == male:  ', round(male_mean, 3))


# Difference (абсолютная разница):
Показывает, на сколько процентов отличается вероятность события между двумя группами.
Пример: global_mean 15%, а в группе с признаком no_partner — 10%, то difference = 15% - 10% = 5% разницы.
Если разница меньше нуля то вероятность ухода клиентов с данным признаком высока
Если разница больше нуля то вероятность ухода клиентов с данным признаком низкая


# Risk Ratio (относительный риск):
Показывает, во сколько раз вероятность события больше в одной группе по сравнению с другой.
Пример: global_mean 15%, а в группе с признаком no_partner — 10%, то risk ratio = global_mean / <признак> = 15% / 10% = 1.5 раза больше риск.
Если разница больше одного то вероятность ухода клиентов с данным признаком высока
Если разница меньше одного то вероятность ухода клиентов с данным признаком низкая

"""

# Например:
# Признаки "есть партнер" и "нет партнера" сравним с global_mean
"""
partner_yes = df_train_full[df_train_full.partner == 'yes'].churn.mean()
print('partner == yes:', round(partner_yes, 3))

partner_no = df_train_full[df_train_full.partner == 'no'].churn.mean()
print('partner == no :', round(partner_no, 3))
 
 #partner == yes: 0.205
 #partner == no : 0.33

print('difference = ', global_mean - round(partner_yes, 3 ))
print('difference = ', global_mean - round(partner_no, 3 ))

#difference =  0.06496805111821088 > 0   
#difference =  -0.06003194888178914 < 0  ==> данная группа признаков выглядит значимой 

print('risk ratio = ', partner_yes / global_mean)
print('risk ratio = ', partner_no / global_mean)
"""
#risk ratio =  0.7594724924338315 < 1
#risk ratio =  1.2216593879412643 > 1 ==> данная группа признаков выглядит значимой конечно самая ли значимая мы можем понять после того как изучим все признаки 


# Конечно для изучения всех мы не будем так долго перечислять все признаки а воспользуемся Pandas и циклом
"""
for c in categorical:
    df_group = df_train_full.groupby(c).churn.agg(['mean', 'count']) #Создание датафрейма > Группировка по категориям из списка "categorical" > найти churn каждой группы в категории и вычислить среднии и кол.во
    df_group['diff'] = global_mean - df_group['mean'] # Вычисление абсолютной разницы
    df_group['risk ratio'] = df_group['mean'] / global_mean # Вычисление относительного риска "Risk ratio"
    print(df_group)
"""

# Mutual Information (Взаимная информация) 
# Mutual information (MI) — это мера, которая показывает, насколько сильно одна переменная "рассказывает" о другой. ( Например знание какой у клиента contract насколько важна чтобы определить churn(уход клиента) )
# Например возьмем пол клиентов, если коэффицент взаимосвязи очень близок к нулю это означает то что знание пола клиента не дает мне никакой информации о том будет ли пользователь уходить или удалит приложение и др.
# Чем выше от нуля это значение тем выше значимость одного признака к другому
from sklearn.metrics import mutual_info_score # mutual_info_score это один из инструментов ск лёрна для вычисления взаимной информации который принимает в качестве двух аргументов: две таблицы с признаками       
# mutual_info_score(df_train_full.churn, df_train_full.contract)  насколько информация о контракте дает понять будет ли коиент уходить или нет
def calculate_mi(series):
    return mutual_info_score(series, df_train_full.churn)

df_mi = df_train_full[categorical].apply(calculate_mi)  # .apply это метод для использования функции который принимает только 1 аргумент, в данном случае принимает функцию calculate_mi(series) где series это аргумент который будет выбран из датафрейма df_train_full[<столбец или столбцы>] в нашем случае df_train_full[categorical]
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')


print(df_mi)

"""
                        MI
contract          0.098320
onlinesecurity    0.063085
techsupport       0.061032
internetservice   0.055868
onlinebackup      0.046923
deviceprotection  0.043453
paymentmethod     0.043210
streamingtv       0.031853
streamingmovies   0.031581
paperlessbilling  0.017589
dependents        0.012346
partner           0.009968
seniorcitizen     0.009410
multiplelines     0.000857
phoneservice      0.000229
gender            0.000117

"""



# Correalation 
#Корреляция показывает, есть ли связь между двумя числовыми признаками и насколько она сильная и направленная.
# Negative correlation:
# -0.2 < r < 0 слабая зависимость или ее отсутствие
# -0.5 < r < -0.2 средняя зависимость
# -1 < r < -0.6 сильная зависимость

# Positive correlation:
# 0 < r < 0.2 слабая зависимость или ее отсутствие
# 0.2 < r < 0.5 средняя зависимость
# 0.6 < r < 1 сильная зависимость

#print(df_train_full[numerical].corrwith(df_train_full.churn).to_frame('correlation')) # .corrwith - вычисление корреляции между значениями из списка numerical и churn
#print(df_train_full.groupby(by='churn')[numerical].mean())

"""
               correlation
tenure            -0.351885 ==> Негативная кореляция если увеличивается время провождения с нашей компанией то уменьшается процент ухода клиента
monthlycharges     0.196805 ==> Позитивная кореляция если увеличивается месячные затраты клиента за наши услуги то увеличивается процент ухода клиента
totalcharges      -0.196353 ==> Негативная кореляция если увеличивается общие затраты клиента за наши услуги то уменьшается процент ухода клиента

          tenure  monthlycharges  totalcharges
churn                                         
0      37.531972       61.176477   2548.021627
1      18.070348       74.521203   1545.689415

"""

# One-hot Encoding

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)
#train_dict = df_train[categorical + numerical].to_dict(orient='records')
#train_dict[0]


#dv.fit(train_dict)

#X_train = dv.transform(train_dict)
#X_train.shape

#val_dict = df_val[categorical + numerical].to_dict(orient='records')
#X_val = dv.transform(val_dict)

#dv.get_feature_names_out()
"""
1. Преобразование обучающего датафрейма (df_train) в список словарей:

train_dict = df_train[categorical + numerical].to_dict(orient='records')
Теперь каждая строка — это словарь например:
train_dict[0]
{'gender': 'male','seniorcitizen': 'partner': 'yes'dependents': 'no','phoneservice': 'yes','multiplelines': 'no','internetservice': 'dsl','onlinesecurity': 'yes','onlinebackup': 'yes','deviceprotection': 'yes','techsupport': 'yes','streamingtv': 'yes','streamingmovies': 'yes','contract': 'two_year','paperlessbilling': 'yes','paymentmethod': 'bank_transfer_(automatic)','tenure': 71,'monthlycharges': 86.1,'totalcharges': 6045.9}

{'gender': 'male', 'seniorcitizen': 0, 'partner': 1, ...} вид информации каждого пользователя

2. Инициализация DictVectorizer

dv = DictVectorizer(sparse=False)
DictVectorizer превращает:

категориальные признаки → в one-hot encoding (0 и 1),

числовые → оставляет как есть.

sparse=False — значит, результат будет обычной NumPy-матрицей.

3. Обучение преобразователя (fit) на тренировочных данных

dv.fit(train_dict)
Сохраняются все категории и признаки, чтобы потом правильно их преобразовывать.

4. Преобразование обучающих данных в матрицу признаков

X_train = dv.transform(train_dict)
Результат — матрица X_train, где категориальные признаки закодированы, а числовые — остались как есть.

5. Повторяем то же самое для валидации

Для валидации просто вызываем transform(), чтобы применить ту же схему, что и для обучения.

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)
### Важно:
Мы НЕ вызываем fit() второй раз!
→ иначе словарь признаков изменится, и модель не сможет корректно обработать данные.
Например:

train_dict:
gender: female, male
после fit() он запомнил категории в порядке:
gender=female, gender = male

Если ты вызвать fit() на val_dict, допустим там только:
gender: male
он "забудет" про female и создаст новый набор признаков:
gender = male

6. Посмотреть все созданные признаки

dv.get_feature_names()

['contract=month-to-month',
 'contract=one_year',
 'contract=two_year',

 'dependents=no',
 'dependents=yes',

 'deviceprotection=no',
 'deviceprotection=no_internet_service',
 'deviceprotection=yes',

 'gender=female',
 'gender=male',

 'internetservice=dsl',
 'internetservice=fiber_optic',
 'internetservice=no',

 'monthlycharges', => numeric

 'multiplelines=no',
 'multiplelines=no_phone_service',
 'multiplelines=yes',

 'onlinebackup=no',
 'onlinebackup=no_internet_service',
 'onlinebackup=yes',

 'onlinesecurity=no',
 'onlinesecurity=no_internet_service',
 'onlinesecurity=yes',

 'paperlessbilling=no',
 'paperlessbilling=yes',

 'partner=no',
 'partner=yes',

 'paymentmethod=bank_transfer_(automatic)',
 'paymentmethod=credit_card_(automatic)',
 'paymentmethod=electronic_check',
 'paymentmethod=mailed_check',

 'phoneservice=no',
 'phoneservice=yes',

 'seniorcitizen',

 'streamingmovies=no',
 'streamingmovies=no_internet_service',
 'streamingmovies=yes',

 'streamingtv=no',
 'streamingtv=no_internet_service',
 'streamingtv=yes',

 'techsupport=no',
 'techsupport=no_internet_service',
 'techsupport=yes',

 'tenure', => numeric

 'totalcharges'] => numeric

"""
# Logistic Regression


# Основная идея Логистическая регрессия — это классификационный алгоритм, который выступает адаптацией линейной регрессии:
# к линейной комбинации признаков (z = w₀ + Σ wᵢ xᵢ) применяется сигмоида: функция, сжимающая z в диапазон [0, 1]. sigmoid = 1/1+np.expm1(-z), чтобы выдавать вероятность события (например, P(churn)).
# Обучение модели
# Вместо метрики MSE используется Log Loss (линейгоровая функция потерь):
# −[ylog(p)+(1−y)log(1−p)]
# Для минимизации используется градиентный спуск (в Scikit‑learn — стохастический SGD).

# Выводы:
# Логистическая регрессия — это линейная модель + сигмоида.
# Используется для бинарной классификации.
# Потери — log‑loss.
# Интерпретация через веса и log-odds делает подход прозрачным и понятным.



# Training Model (Logistic Regression)
# from sklearn.linear_model import LogisticRegression импортировали в начале
# model = LogisticRegression(solver='liblinear', random_state=1)
# model.fit(X_train, y_train) # Модель логистической регрессии не преобразует данные, а учится на них. Поэтому у неё есть только fit(), чтобы запомнить зависимость между признаками и ответами.
# Главное различие:
"""
     DictVectorizer	                      LogisticRegression
.fit() — ищет признаки	               .fit() — обучает модель
.transform() — преобразует данные	   нет .transform()
Работает с признаками	               Работает с признаками и целями
НЕ делает предсказаний	               Делает предсказания (predict, predict_proba)

"""
"""
LogisticRegression(random_state=1, solver='liblinear')
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict) 
model.predict_proba(X_val)
y_pred = model.predict_proba(X_val)[:, 1]
# y_pred = model.predict_proba(X_val) дает матрицу из двух колонок которые вероятности оставания клиента и ухода мы берем ухода то есть вторую колонку добавляя [:, 1]
churn = y_pred > 0.5
(y_val == churn).mean() # ==> 0.8016129032258065
model.intercept_[0]  # w0 = -0.12198811467233629
model.coef_[0].round(3)  # ===> Веса(w)
"""
"""
шаг 1: Создаём модель логистической регрессии

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', random_state=1)

# LogisticRegression — это алгоритм, который будет учиться на данных.

# solver='liblinear' — быстрый оптимизатор для маленьких датасетов

# random_state=1 — чтобы каждый раз результат был одинаковым.

Шаг 2: Обучаем модель на примерах

model.fit(X_train, y_train)
X_train — это таблица признаков (например, возраст, марка машины и т.д.).

y_train — это ответы (ушёл клиент или нет, 1 или 0).

После этого шага модель "запоминает закономерности" между признаками и результатами.

Шаг 3: Готовим данные для проверки (валидации)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)
Переводим валидационные данные (df_val) в список словарей.

Превращаем словари в числовую таблицу, как делали для обучения.

Важно: структура признаков должна быть такой же, как в обучении.

Шаг 4: Делаем предсказания

model.predict_proba(X_val)
Эта команда выдаёт вероятности для каждого клиента:
Пример:

[[0.7, 0.3],
 [0.2, 0.8],
 [0.6, 0.4]]
Это значит:

Первый клиент: 70% не уйдёт (0), 30% уйдёт (1)

Второй клиент: 20% не уйдёт, 80% уйдёт

Шаг 5: Берём именно вероятность "уйдёт" (класс 1)

y_pred = model.predict_proba(X_val)[:, 1]
Теперь у нас есть просто список вероятностей ухода:

[0.3, 0.8, 0.4, ...]

Шаг 6: Переводим вероятности в 0 и 1

Пример:
y_pred = [0.12, 0.95, 0.67, 0.3, 0.81]

churn = y_pred > 0.5
# Результат:
# [False, True, True, False, True]

Если вероятность больше 50%, считаем, что клиент уйдёт (True или 1).

Шаг 7: Сравниваем с реальными ответами

(y_val == churn).mean()
Сравниваем реальные ответы (y_val) и предсказания (churn)

Считаем, сколько предсказаний правильные (доля правильных ответов = accuracy)

"""

# Model Interpretation (By using small subset)

#dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))  сопоставляет по порядку признаки из значения и их веса

""" ==>

{'contract=month-to-month': 0.563,
 'contract=one_year': -0.086,  
 'contract=two_year': -0.599,
 'dependents=no': -0.03,
 'dependents=yes': -0.092, ........}
 


subset = ['contract', 'tenure', 'totalcharges']  # сабсет признаков для объяснения работы модели

train_dict_small = df_train[subset].to_dict(orient='records')  # Создание словаря с признаками из сабсета всех клиентов типа: {'contract': 'one_year', 'tenure': 55, 'totalcharges': 5656.75}
dv_small = DictVectorizer(sparse=False)
dv_small.fit(train_dict_small) #  fit(): Находит все уникальные категориальные значения (категории). Запоминает клиентские признаки и порядок, в котором их потом будет кодировать.

X_small_train = dv_small.transform(train_dict_small) # DictVectorizer превращает: категориальные признаки → в one-hot encoding (0 и 1), числовые → оставляет как есть. sparse=False — значит, результат будет обычной NumPy-матрицей.
dv_small.get_feature_names_out() # Выведет признаки и их значения ['contract=month-to-month 'contract=one_year', 'contract=two_year', 'tenure', 'totalcharges']

model_small = LogisticRegression(solver='liblinear', random_state=1) # Вызывание модели и указание параметров
model_small.fit(X_small_train, y_train)  # Обучение модели
LogisticRegression(random_state=1, solver='liblinear') 
model_small.intercept_[0] # Нулевой вес (w0)

dict(zip(dv_small.get_feature_names_out(), model_small.coef_[0].round(3))) # Выведение значений признаков и его веса по порядку. zip() используется для того чтобы поочередно выбирать 1 элемент первого списка и первый элемент второго списка и так по порядку
{'contract=month-to-month': 0.866,
 'contract=one_year': -0.327,
 'contract=two_year': -1.117,
 'tenure': -0.094,
 'totalcharges': 0.001}
val_dict_small = df_val[subset].to_dict(orient='records') # Конвертирование валидационных данных в словарь
X_small_val = dv_small.transform(val_dict_small)  # DictVectorizer превращает: категориальные признаки → в one-hot encoding (0 и 1), числовые → оставляет как есть. sparse=False — значит, результат будет обычной NumPy-матрицей.
y_pred_small = model_small.predict_proba(X_small_val)[:, 1] # Предсказывание значений (y) и выбор второго стобца где собраны вероятности churn (второй столбец)


"""

# Using the model

dv = DictVectorizer(sparse=False)
train_full_dict = df_train_full[categorical + numerical].to_dict(orient='records')
X_train_full = dv.fit_transform(train_full_dict) 
y_train_full = df_train_full.churn.values
model = LogisticRegression(solver='liblinear',random_state=1)
model.fit(X_train_full,y_train_full)



dicts_test = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(dicts_test)

y_pred = model.predict_proba(X_test)[:, 1]

churn_decision = (y_pred >= 0.5)
print((churn_decision == y_test).mean())

# Тест модели на клиентах
customer_1 = {
    'customerid': '8879-zkjof',
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 41,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'yes',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 79.85,
    'totalcharges': 3320.75,
}
X_test_1 = dv.transform([customer_1])
# X_test_1[0] ==> [0.0, 1.0, 0.0, 1.0, ... 41.0, 3320.75]
print('вероятность churn 1 клиента')
y_customer_1 = model.predict_proba(X_test_1)[0, 1]
print(y_customer_1)


customer_2 = {
    'gender': 'female',
    'seniorcitizen': 1,
    'partner': 'no',
    'dependents': 'no',
    'phoneservice': 'yes',
    'multiplelines': 'yes',
    'internetservice': 'fiber_optic',
    'onlinesecurity': 'no',
    'onlinebackup': 'no',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'yes',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 85.7,
    'totalcharges': 85.7
}

X_test_2 = dv.transform([customer_2])
y_customer_2 = model.predict_proba(X_test_2)[0, 1] # вероятность черна (2 столбец)
print('вероятность churn 2 клиента') 
print(y_customer_2)
