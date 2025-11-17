###1. Скачуємо бібліотеки
#pip install tensorflow pandas numpy matplotlib scikit-learn
import pandas as pd #працюємо з файлами csv таблицями
import numpy as np #математичні операції
import tensorflow as tf #Створює нейронку
#keras - частина тф, яка створює шари в нейронці шар-група нейронів одного рівня
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder #переводить назви в числа, для того щоб нейронка розуміла, що це саме ця фігура
import matplotlib.pyplot as plt #для побудови візуалізації даних, графіків


###2. Зчитуємо інформацію з csv таблиці
df = pd.read_csv("data/figures.csv")#datafile
# print(df.head())


###3. Перетворюємо значення фігурам в числа
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label']) #створюємо новий стовпець в таблиці. Значення - перетворений рядок label

X = df[['area', 'perimeter', 'corners']] #додаємо наші параметри в ознаки
y = df[['label_enc']] #додаємо вже перетворені числа для фігур 20рядок) в мітки


###4. Створення моделі
#Послідовно з`єднанні шари, табличні дані - тип нейронки
# model = keras.Sequential(layers.Dense(8, activation = "relu")) #Три аргументи - шари. 1-кількість нейронів(8) 2-активація (кортеж параметрів)
model = keras.Sequential([
    layers.Dense(8, activation = 'relu', input_shape = (3,)),
    layers.Dense(8, activation = 'relu'),
    layers.Dense(8, activation = 'softmax')
])


###.5 Компаляція моделі (визначення навчання)
model.compile(optimizer = 'adam', loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
#1 - оптимізація, adam - визначає алгоритм для навчання, 2 -втрати, 3 - точність


###.6 Навчання
history = model.fit(X, y, epochs = 300, verbose = 0) #Останній аргумент необов`язковий. Не виводить аналіз

#Графіки: 1 - втрата, 2 - точність
plt.plot(history.history['loss'], label = "Втрати")
plt.plot(history.history['accuracy'], label = "Точність")
#Підписуємо вісь x та y
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title('Процес навчання моделі')
plt.legend()
plt.show()


###.7 Тестування
test = np.array(df[25, 20, 0])
predict = model.predict(test) #Отримуємо ймовірність

print(f'Імовірність кожного класу {predict}')
print(f'Модель визначила {encoder.inverse_transform([np.argmax(predict)])}')