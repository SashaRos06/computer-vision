import cv2
import numpy as np

image = cv2.imread('images/photo_1.jpg')
image = cv2.resize(image, (image.shape[1] // 6, image.shape[0] // 6))
# print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 240, 200)
kernel = np.ones((5, 5), np.uint8)
image = cv2.dilate(image, kernel, iterations = 1)
image = cv2.erode(image, kernel, iterations = 1)

cv2.imshow('photo', image)


# image = cv2.imread('images/gmail.jpg')
# image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 220, 120)
# kernel = np.ones((5, 5), np.uint8)
# image = cv2.dilate(image, kernel, iterations = 1)
# image = cv2.erode(image, kernel, iterations = 1)



cv2.imshow('photo', image)
cv2.waitKey(0)
cv2.destroyAllWindows()