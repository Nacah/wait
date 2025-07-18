import numpy as np
# множество это матрица 1 го измерения [1,2,3,4] и т.д.
# np.zeros(10) # матрица c одним измерением из нулей, внутрь пишется кол во нулей
# np.ones(10) # матрица из однёрок внутрь пишется кол во однёрок
# np.full(кол во чисел которые хочешь внутри матрицы, число которое хочешь) # создает матрицу(множество) из числа которого хочешь
# np.array([1,2,3,4,5,6]) делает матрицу из списка


# Generating Ranges of Numbers
# Numpy provides functions for generating arrays of sequential numbers. For example:
# range_array = np.arange(10)  # Creates an array from 0 to 9
 


# Creating Arrays with Linear Spacing

# np.linspace() creates arrays with evenly spaced numbers within a specified range:
# linspace_array = np.linspace(0, 1, 11)  # Creates 11 numbers from 0 to 1



# Multi-dimensional Arrays
# Numpy can handle multi-dimensional arrays, often referred to as matrices. Here are some examples:

# np.type of matrix(rows, columns)
# zeros_matrix = np.zeros((5, 2))
# ones_matrix = np.ones((5, 2))
# constant_matrix = np.full((5, 2), 3)



# Indexing and Slicing Arrays
# Like Python lists, you can access elements in Numpy arrays using indexing and slicing. For two-dimensional arrays:

# arr = np.array([[2, 3, 4], [4, 5, 6]])   ===>  [[2 3 4]
#                                                 [4 5 6]]
# first_row = arr[0]      # Gets the first row  ==> [2,3,4]
# first_col = arr[:, 0]  # Gets the first column ==> [2,4]



# Generating Random Arrays
# Numpy can create arrays filled with random numbers. To ensure reproducibility, you can set a seed using np.random.seed():

# np.random.seed(2)  # Set the seed зафиксировать рандомно созданные числа
# random_array = np.random.rand(5, 2)  # Generates random numbers between 0 and 1
# random_array = 100 * np.random.rand(5, 2)  # Generates random numbers between 0 and 100
# For random numbers from a normal distribution or integers within a range:
# normal_distribution = np.random.randn(5, 2)
# random_integers = np.random.randint(low=0, high=100, size=(5, 2))



# Array Operations
# Numpy excels in performing mathematical operations on arrays efficiently.

# Element-wise Operations
# You can perform operations on entire arrays element by element:
# for exampla arr =[[2 3 4]
#                   [4 5 6]]
# arr = arr + 1   # Adds 1 to each element [[3 4 5]
#                                           [6 7 8]] 
# arr = arr * 2   # Multiplies each element by 2
# Similar operations for division and exponentiation



# Element-wise Operations with Two Arrays
# You can also perform operations between two arrays of the same shape:

# arr1 = np.ones(4)
# arr2 = np.full(4, 3)
# result = arr1 + arr2  # Element-wise addition
# result = arr1 / arr2  # Element-wise division
# Comparison Operations
# You can perform element-wise comparisons and create boolean arrays:

# arr = np.array([1, 2, 3, 4])
# greater_than_2 = arr > 2  # Produces [False, False, True, True]



# Selecting Elements Based on Conditions
# You can create subarrays based on certain conditions:

# selected_elements = arr[arr > 1]  # Gets arr matrix with elements greater than 1
# Summary Operations
# Numpy provides functions for summarizing array data:

# min_value = arr.min()    # Minimum value
# max_value = arr.max()    # Maximum value
# sum_value = arr.sum()    # Sum of all elements
# mean_value = arr.mean()  # Mean (average) value
# std_deviation = arr.std()  # Standard deviation



""" ###LINEAR ALGEBRA REFRESHER###
Vector operations
u = np.array([2, 7, 5, 6])
v = np.array([3, 4, 8, 6])

# addition 
u + v

# subtraction 
u - v

# scalar multiplication 
2 * v



Multiplication

Vector-vector multiplication
def vector_vector_multiplication(u, v):
    assert u.shape[0] == v.shape[0]
    
    n = u.shape[0] #.shape() показывает размер матрицы так как у нас вектор то размер будет (4,1) 
    но!!! так как в пайтоне векторы показываются ввиде строк выводить просто 4, так как размера вектора u = 4
    
    result = 0.0

    for i in range(n):
        result = result + u[i] * v[i]
    
    return result
Matrix-vector multiplication
def matrix_vector_multiplication(U, v):
    assert U.shape[1] == v.shape[0] #Здесь важно, мы хотим проверить чтобы совпадало кол.во колонок в матрице U и
    кол.во строк в векторе v потому что умножение тогда не получится, поэтому мы сравниваем колонки U.shape[1] и строки v.shape[0]

    
    num_rows = U.shape[0]
    
    result = np.zeros(num_rows) #делаем матрицу сосотоящую из нулей в том же формате что и будет наш ответ тоесть вектором состоящим из трёх строк

    
    for i in range(num_rows):
        result[i] = vector_vector_multiplication(U[i], v) #Векторное умножение между каждой строкой матрицы U и всем вектором v 
                                                           Считай что ты берешь строки матрицы повариваешь направо чтобы они стали вертикально
                                                           как вектор v, и умножаешь каждый элемент первой строки с каждым элементом вектора и складываешь результаты это будет первым элементым вектора Uv 
                                                           потом со второй строкой матрицы делаешь тоже самое и с третьей тоже и получится вектор с 3 строками     
    return result  
Matrix-matrix multiplication
def matrix_matrix_multiplication(U, V):
    assert U.shape[1] == V.shape[0] тоже самое правило кол во колонок первой матрицы должна совпадать с кол.вом строк второй матрицы
    
    num_rows = U.shape[0]
    num_cols = V.shape[1]
    
    result = np.zeros((num_rows, num_cols)) формируем нашей матрицу размер если мы проверяли колонки первой матрицы со строками второй матрицы значит 
                                            размер конечной матрицы будет наоборот количеством строк первой матрицы и количеством колонок второй матрицы
    
    for i in range(num_cols):
        vi = V[:, i]
        Uvi = matrix_vector_multiplication(U, vi)
        result[:, i] = Uvi
    
    return result
Identity matrix
I = np.eye(3)
Inverse
V = np.array([
    [1, 1, 2],
    [0, 0.5, 1], 
    [0, 2, 1],
])
inv = np.linalg.inv(V)

"""

""" PANDAS
data = [
    ['Nissan', 'Stanza', 1991, 138, 4, 'MANUAL', 'sedan', 2000],
    ['Hyundai', 'Sonata', 2017, None, 4, 'AUTOMATIC', 'Sedan', 27150],
    ['Lotus', 'Elise', 2010, 218, 4, 'MANUAL', 'convertible', 54990],
    ['GMC', 'Acadia',  2017, 194, 4, 'AUTOMATIC', '4dr SUV', 34450],
    ['Nissan', 'Frontier', 2017, 261, 6, 'MANUAL', 'Pickup', 32340],
]

columns = [
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders',
    'Transmission Type', 'Vehicle_Style', 'MSRP'
]
 
df = pd.DataFrame(data, columns = columns)
print(df) ===>                Make     Model  Year     Engine HP      Engine Cylinders  Transmission   Type Vehicle_Style   MSRP
                         0   Nissan    Stanza  1991      138.0                 4            MANUAL         sedan            2000
                         1  Hyundai    Sonata  2017        NaN                 4         AUTOMATIC         Sedan           27150
                         2    Lotus     Elise  2010      218.0                 4            MANUAL   convertible           54990
                         3      GMC    Acadia  2017      194.0                 4         AUTOMATIC       4dr SUV           34450
                         4   Nissan  Frontier  2017      261.0                 6            MANUAL        Pickup           32340
	
df.head(n=2) ===> отображение первых двух записей в таблице  ===>      Make     Model  Year     Engine HP      Engine Cylinders  Transmission Type     Vehicle_Style       MSRP
                                                                  0   Nissan    Stanza  1991      138.0                 4                MANUAL           sedan            2000
                                                                  1  Hyundai    Sonata  2017        NaN                 4               AUTOMATIC         Sedan           27150
###чтобы получить какую то колонку из таблицы 
df.название колонки: df.Make ===>                     Make     
или df["название колонки"]: df["Make"] ===>       0  Nissan    
                                                  1  Hyundai    
                                                  2   Lotus     
                                                  3    GMC
                                                  4   Nissan
df[["Make", "Model","MSRP"]] # Отображение нескольких колонок
make_series = pd.Series(df['Make']) создание серии/колонки состоящей тольки из марок машин
df['id'] = [10, 20, 30, 40, 50] # добавление новой колонки
del df['id'] ** удаление
df.index
# Output: RangeIndex(start=0, stop=5, step=1)

***Вытаскиваем элементы датафреймов
df.loc[1] элемент с индексом 1 то есть вторая машина в таблице
# Output:
# Make                   Hyundai
# Model                   Sonata
# Year                      2017
# Engine HP                  NaN
# Engine Cylinders             4
# Transmission Type    AUTOMATIC
# Vehicle_Style            Sedan
# MSRP                     27150
# Name: 1, dtype: object
df.index = ['a', 'b', 'c', 'd', 'e'] ** смена индексов **** После смены индексов df.loc[1] такое обращение даст ошибку 
                                                            но все еще можно вызывать элементы по позиционному индексу df.iloc[[1, 2, 4]]
df.reset_index() ** перезапуск индексов
df.reset_index(drop = True) ** если не нужны буквенные индексы которые мы установили
### Операции с колонками/элементами
df['Engine HP'] / 100
# Output:
# 0    1.38
# 1     NaN
# 2    2.18
# 3    1.94
# 4    2.61
# Name: Engine HP, dtype: float64

////ФИЛЬТРАЦИЯ ДАННЫХ////
df['Year'] >= 2015 ===> таблица с boolean значениями true если после 2015 машина была выпущена false если до
df[df['Year'] >= 2015] **выдаст таблицу только с машинами 2015 года и выше

df[(df.Make == 'Nissan') & (df.Year > 2015)]  ** комбинация условий

df['Vehicle_Style'].str.lower() **уменьшение букв в колонке vehicle style

A typical pre-processing step when you work with text is to replace all spaces with underscores.

'machine learning zoomcamp'.replace(' ','_')
# Output: 'machine_learning_zoomcamp'

df['Vehicle_Style'].str.replace(' ','_')
# Output:
# 0          sedan
# 1          Sedan
# 2    convertible
# 3        4dr_SUV
# 4         Pickup
# Name: Vehicle_Style, dtype: object
 
# Both operations can be chained
df['Vehicle_Style'] = df['Vehicle_Style'].str.lower().str.replace(' ','_')  ** сохранение изменений


///Числовые операции
df.MSRP.mean()

df.MSRP.describe()
# Output:
# count        5.000000
# mean     30186.000000
# std      18985.044904
# min       2000.000000
# 25%      27150.000000
# 50%      32340.000000
# 75%      34450.000000
# max      54990.000000
# Name: MSRP, dtype: float64

df.describe().round(2) ** вычисление характеристик выше для всей таблицы  **.round окрушление до двух

# Returns the number of unique values of column Make
df.Make.nunique()
# Output: 4
 
# Returns the number of unique values for all columns of the DataFrame
df.nunique()
# Output:
# Make                 4
# Model                5
# Year                 3
# Engine HP            4
# Engine Cylinders     2
# Transmission Type    2
# Vehicle_Style        4
# MSRP                 5
# dtype: int64


///Реализация пустых зон
Missing values can make our lives more difficult. That’s why it makes sense to take a look at this problem. The function isnull() returns true for each value/cell that is missing.

df.isnull()

The representation is very confusing therefor it’s more useful to sums up the number of missing values for each column.

df.isnull().sum()
# Output:
# Make                 0
# Model                0
# Year                 0
# Engine HP            1
# Engine Cylinders     0
# Transmission Type    0
# Vehicle_Style        0
# MSRP                 0
# dtype: int64

///Группировка///
df.groupby('Transmission Type').MSRP.mean()
# Output:
# Transmission Type
# AUTOMATIC    30800.000000
# MANUAL       29776.666667
# Name: MSRP, dtype: float64

Getting the NumPy arrays
Sometimes it is necessary to convert a Pandas DataFrame back to the underlying NumPy array.

df.MSRP.values
# Output:
# array([ 2000, 27150, 54990, 34450, 32340])


///Конвертирование датафрейма в словарь///
df.to_dict(orient='records')
# Output:
# [{'Make': 'Nissan',
#  'Model': 'Stanza',
#  'Year': 1991,
#  'Engine HP': 138.0,
#  'Engine Cylinders': 4,
#  'Transmission Type': 'MANUAL',
.........
.........
.........

""" 
X_learning_normal_equation= [
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
X_learning_normal_equation_array = np.array(X_learning_normal_equation)
y = [100, 200, 150, 250, 100, 200, 150, 250, 120]
def train_linear_regression(X, y):

    ones = np.ones(X.shape[0])  # Создание столбца нулевых признаков из единиц

    X = np.column_stack([ones, X])  # Добавление столбца нулевых признаков из единиц

    XTX = X.T.dot(X) # Создание gram matrix транспонированная матрица умноженная на оригинальную матрицу

    XTX_inv = np.linalg.inv(XTX) # Инвертирование gram matrix

    w = XTX_inv.dot(X.T).dot(y) # Вычисление весов
    
    return w[0], w[1:] # Нулевой вес и весы признаков
print(train_linear_regression(X_learning_normal_equation_array,y))
