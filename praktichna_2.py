import cv2
import numpy as np

img = cv2.imread('images/shapes.jpg')
img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
img1, img2, img3, img4 = img.copy(), img.copy(), img.copy(), img.copy()
img1, img2, img3, img4 = cv2.GaussianBlur(img1, (3, 3), 1), cv2.GaussianBlur(img2, (3, 3), 1), cv2.GaussianBlur(img3, (3, 3), 1), cv2.GaussianBlur(img4, (3, 3), 1)





img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
lower_1 = np.array([107, 238, 163])
upper_1 = np.array([169, 255, 255])
mask_1 = cv2.inRange(img1, lower_1, upper_1)
img1 = cv2.bitwise_and(img1, img1, mask=mask_1)


img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
lower_2 = np.array([43, 80, 87])
upper_2 = np.array([84, 255, 215])
mask_2 = cv2.inRange(img2, lower_2, upper_2)
img2 = cv2.bitwise_and(img2, img2, mask=mask_2)

img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
lower_3 = np.array([117, 0, 0])
upper_3 = np.array([179, 255, 151])
mask_3 = cv2.inRange(img3, lower_3, upper_3)
img3 = cv2.bitwise_and(img3, img3, mask=mask_3)


img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)
lower_4 = np.array([0, 7, 99])
upper_4 = np.array([49, 248, 255])
mask_4 = cv2.inRange(img4, lower_4, upper_4)
img4 = cv2.bitwise_and(img4, img4, mask=mask_4)


mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_green)
# cv2.imshow("Image", img1)
# cv2.imshow("Imag1", img2)
# cv2.imshow("Imag2", img3)
# cv2.imshow("Imag3", img4)
cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

