import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image

from CW_less10 import prediction

###1. Завантажуємо файли #batch size - скільки зображень бере для навчання
train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train',
                                                               image_size = (128, 128),
                                                               batch_size = 30,
                                                               label_mode = 'categorical')
test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test',
                                                               image_size = (128, 128),
                                                               batch_size = 30,
                                                               label_mode = 'categorical')


###.2 Нормалізація зображень для нейронки
normalization_layer = layers.Rescaling(1./255) #1./255 - переводить в нулики та одинички
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = models.Sequential()

#Прості ознаки - краї, лінії
model.add(layers.Conv2D(
    filters = 32,                 #Кількість фільтрів
    kernel_size = (3, 3),         #розмір фільтра
    activation='relu',            #функція активації
    input_shape=(128, 128, 3)     #форма вхіждного зображення
))
#Складні ознаки
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
#Донавчаємо
model.add(layers.MaxPooling2D(pool_size = (2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2, 2)))

#Передаємо результати в шари
model.add(layers.Flatten())

model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(3, activation = 'softmax'))


###3. Компіляція моделі
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


###.4 Навчання моделі
history = model.fit(train_ds, epochs = 10, validation_data = test_ds,verbose = 0)
test_loss, test_acc = model.evaluate(test_ds)
print(f'Правдивість: {test_acc}')


###.5 Перевірка
class_name = ["cars", "cats", "dogs"]

img = image.load_img('data/test_img.jpg', target_size = (128, 128))
img_array = image.img_to_array(img)
#Нормалізує зображення
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, 0) #Зображення подається- Значення, розмір, кількість кольорових шарів

#Робимо прогноз
predict = model.predict(img_array)
predict_index = np.argmax(predict[0]) #Для визначення класу кіт, собака


###.6 Виведення результатів
print(f'Імовірність по класам: {predict[0]}')
print(f'Модель визначила: {class_name[predict_index]}')