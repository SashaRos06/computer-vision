import cv2
import numpy as np
image = cv2.imread('images/less4.jpg')
image = cv2.resize(image, (image.shape[1] // 6, image.shape[0] // 6))
cv2.putText(image, "(123; 50)x(253; 220)", (128, 40), cv2.FONT_HERSHEY_DUPLEX, 0.35, (14, 50, 207), 1)
cv2.rectangle(image, (123, 50), (253, 220), (14, 50, 207), 2)
cv2.putText(image, "(270; 89)x(385; 250)", (265, 79), cv2.FONT_HERSHEY_DUPLEX, 0.35, (14, 50, 207), 1)
cv2.rectangle(image, (270, 89), (385, 250), (14, 50, 207), 2)
cv2.putText(image, "(415; 85)x(540; 260)", (415, 75), cv2.FONT_HERSHEY_DUPLEX, 0.35, (14, 50, 207), 1)
cv2.rectangle(image, (415, 85), (540, 260), (14, 50, 207), 2)



cv2.imshow('photo', image)
cv2.waitKey(0)
cv2.destroyAllWindows()