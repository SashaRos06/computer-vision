#Створення нейронки, яка буде розпізнавати кольори та фігури
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#####1 - створимо функцію для генерації простих фігур
def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8) #полотно
    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == "triangle":
        points = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img


#####2 - формуємо набори даних
X = [] #список ознак
y = [] #список міток, правильні відповіді, які ми йому задамо. Пояснюємо - це ця фігура, то й то колір

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0)
}
shapes = ["circle", "square", "triangle"]

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape) #приймає два аргументи
            mean_color = cv2.mean(img)[:3]#повертає середнє значення наших кольорів 1-значенн b, 2-значення g, alpha
            features = [mean_color[0], mean_color[1], mean_color[2]]
            X.append(features) #Додадає кольори в ознаки
            y.append(f'{color_name}_{shape}')


#####3 - розділяємо дані за наступною пропорцією 70%-для навчання, 30%-для перевірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)#Збереження однакової пропорції між тренуванням та навчанням)#x-train - ознаки для навчання, x-test - ознаки для перевірки


#####4 - навчаємо саму модель (from sklearn.neighbors import KNeighborsClassifier)
#Вчиться порівнювати об`єкти за лівим кутом і обирає певні ознаки, за якими буде розпізнаватися фігура і колір
model = KNeighborsClassifier(n_neighbors=3) #ставимо непарні числа 4 (2, 2) 6 (3,3) 50%
model.fit(X_train, y_train)


#####5 - перевіряємо точність
accuracy = model.score(X_test, y_test)
print(f'Точність моделі: {round(accuracy * 100, 2)}%')

test_image = generate_image((0, 240, 20), "square")
mean_color = cv2.mean(test_image)[:3]
prediction = model.predict([mean_color])
print(f'Передбачення: {prediction[0]}')



cv2.imshow("img", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()