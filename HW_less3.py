import cv2
import numpy as np
img = cv2.imread('images/photo.jpg')

cv2.waitKey(0)

cv2.rectangle(img, (200, 20), (350, 200), (14, 50, 207), 2) #207, 50, 14
print(img.shape)
cv2.putText(img, "Oleksandra Rosokhata", (169, 235), cv2.FONT_HERSHEY_DUPLEX, 0.6, (14, 50, 207), 1)


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()