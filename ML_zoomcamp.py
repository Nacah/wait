import pandas as pd
import numpy as np


df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')  #Работа с названиями колонок, уменшили буквы и поставили нижний пробел для читабельности
strings = list(df.dtypes[df.dtypes == "object"].index)  # обозначили какие колонки меют строковые значения
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')  #Работа с элементами колонок, уменьшили буквы и поставили нижний пробел для читабельности

df.head()