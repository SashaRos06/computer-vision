import cv2
import numpy as np


img = np.zeros((512, 400, 3), np.uint8)
#rgb = bgr
# img[:] = 78, 181, 54 #54, 181, 78

# img[100:150, 200:280] = 78, 181, 54 Просто заливає одним кольором певний фрагмент
#Спочатку координати y, а потім x

#Квадрат
cv2.rectangle(img, (100, 100), (200, 200), (78, 181, 54), 2) #Перші два аргументи - координати лівої верхньої та правої нижньої. 3 - кольори 4 - товщина лінії



#Лінія
cv2.line(img, (100, 100), (200, 200), (78, 181, 54), 2) #Перший аргумент - початок лініїЮ другий - кінець
cv2.line(img, (200, 100), (100, 200), (78, 181, 54), 2)

# cv2.line(img, (0, 256), (400, 256), (78, 181, 54), 2)
cv2.line(img, (0, img.shape[0]//2), (img.shape[0], img.shape[0]//2), (78, 181, 54), 2)
cv2.line(img, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (78, 181, 54), 2)



#Коло
cv2.circle(img, (200, 200), 20, (78, 181, 54), 2)



#Текст
cv2.putText(img, "Sasha", (200, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (78, 181, 54), 2)

# print(img.shape)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()