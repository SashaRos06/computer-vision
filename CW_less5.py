import cv2
import numpy as np

image = cv2.imread("images/animal.jpg")
image_copy = image.copy()
image = cv2.GaussianBlur(image, (3, 3), 0)

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([70, 14, 40]) #Мінімальний поріг зображення
upper = np.array([116, 153, 255]) #Максимальний поріг зображення

mask = cv2.inRange(image, lower, upper)
#Накладаємо маску на наше зображення
image = cv2.bitwise_and(image, image, mask = mask)
#Перебираємо контури
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        periment = cv2.arcLength(cnt, True) #True - замкнутий контур
        #Момент контур. Описуємо форму фігури, розміри
        M = cv2.moments(cnt) #М - показники
        #Вираховуємо центр мас. Центроїд - середня позиція контура
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        #Обмежувальний контур
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2) #Допомагає відрізняти співвідношення сторін (прямокутник чи квадрат)
        #Міра округлості об`єкта
        compactness = round((4 * np.pi * area) / (periment ** 2), 2)

        approx = cv2.approxPolyDP(cnt, 0.02 * periment, True) #Кількість вершин
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "square"
        elif len(approx) > 8:
            shape = "oval"
        else:
            shape = "another"
        cv2.drawContours(image_copy, [cnt], -1, (255, 255, 255), 2)
        cv2.circle(image_copy, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(image_copy, f'shape: {shape}', (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image_copy, f'A:{int(area)}, P:{int(periment)}', (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image_copy, f'AR:{aspect_ratio}, C:{compactness}', (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("mask", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()