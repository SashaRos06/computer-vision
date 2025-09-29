import cv2
import numpy as np

###########################################################Зображення
# image = cv2.imread('images/flower.jpg')
# image = cv2.resize(image, (500, 250))
# image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))
# print(image.shape)
# # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) #Для обертання зображення
# # image = cv2.flip(image, 0) #0-горизонтально 1-вертикально
# # image = cv2.GaussianBlur(image, (7, 7), 7) #Можна ставити тільки непарні значення
#
# #Щоб оптимізувати відображення для комп`ютера
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 220, 430)
# # image = cv2.dilate(image, None, iterations = 1) #2-масив значень пікселів(kernel) 3-кількість ітерацій
# kernel = np.ones((5, 5), np.uint8) #Числа більше нуля - unit8
# image = cv2.dilate(image, kernel, iterations = 1)
# image = cv2.erode(image, kernel, iterations = 1)
#
# print(image.shape)
# cv2.imshow('flower', image)
# cv2.imshow('flowers', image[0:250, 0:160])



#######################################################Відео
# video = cv2.VideoCapture('video/kangaroo.mp4')
video = cv2.VideoCapture(0) #вебкамера

while True:
    success, frame = video.read()
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break



cv2.waitKey(0)
