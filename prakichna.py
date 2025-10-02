import cv2
import numpy as np
canv = np.zeros((400, 600, 3), np.uint8)
canv[:] = 252, 247, 235 #235, 247, 252
cv2.line(canv, (15, 15), (585, 15), (242, 221, 170), 2)
cv2.line(canv, (15, 15), (15, 385), (242, 221, 170), 2)
cv2.line(canv, (585, 15), (585, 385), (242, 221, 170), 2)
cv2.line(canv, (15, 385), (585, 385), (242, 221, 170), 2)

image = cv2.imread('images/photo.jpg')
image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
# print(image.shape)

y, x = 70, 40
h = image.shape[0]
w = image.shape[1]
canv[y:y + h, x:x + w] = image


cv2.putText(canv, "Sasha Rosokhata", (210, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (31, 4, 22), 2) #22, 4, 31
cv2.putText(canv, "Computer vision student", (210, 130), cv2.FONT_HERSHEY_COMPLEX, 0.5, (99, 93, 97), 1) #97, 93, 99

cv2.putText(canv, "Email: sashunro@gmail.com", (210, 190), cv2.FONT_HERSHEY_DUPLEX, 0.7, (31, 4, 22), 1)
cv2.putText(canv, "Phone: +38 068 550 72 10", (210, 210), cv2.FONT_HERSHEY_DUPLEX, 0.7, (31, 4, 22), 1)
cv2.putText(canv, "06/03/2010", (210, 230), cv2.FONT_HERSHEY_DUPLEX, 0.7, (31, 4, 22), 1)

cv2.putText(canv, "OPENCV BUSINESS CARD", (120, 320), cv2.FONT_HERSHEY_COMPLEX, 0.7, (31, 4, 22), 2)

qr = cv2.imread('images/qr.jpeg')
qr = cv2.resize(qr, (image.shape[1] // 2, image.shape[0] // 2))
y1, x1 = 250, 460
h = qr.shape[0]
w = qr.shape[1]
canv[y1:y1 + h, x1:x1 + w] = qr

cv2.imshow('image', canv)
cv2.imwrite("business_card.png", canv)
cv2.waitKey(0)
cv2.destroyAllWindows()