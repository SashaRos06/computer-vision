import cv2
import numpy as np
img = cv2.imread('images/figures.png')
img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
img1 = cv2.GaussianBlur(img, (3, 3), 0)
# lower_triangle, lower_oval, lower_square, lower_star = np.array([139, 0, 0]), np.array([0, 9, 0]), np.array([66, 0, 0]), np.array([0, 206, 80])
# upper_triangle, upper_oval, upper_square, upper_star = np.array([179, 255, 255]), np.array([72, 114, 80]), np.array([84, 255, 255]), np.array([179, 255, 142])
lower_triangle = np.array([140, 0, 0])
upper_triangle = np.array([179, 255, 255])
mask_triangle = cv2.inRange(img1, lower_triangle, upper_triangle)
img1 = cv2.bitwise_and(img1, img1, mask = mask_triangle)
contours_t, _ = cv2.findContours(mask_triangle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt_t in contours_t:
    area_t = cv2.contourArea(cnt_t)
    if area_t > 100:
        periment_t = cv2.arcLength(cnt_t, True)
        M_t = cv2.moments(cnt_t)
        if M_t["m00"] != 0:
            cx_t = int(M_t["m10"] / M_t["m00"])
            cy_t = int(M_t["m01"] / M_t["m00"])
        x_t, y_t, w_t, h_t = cv2.boundingRect(cnt_t)
        aspect_ratio_t = round(w_t / h_t, 2)
        compactness_t = round((4 * np.pi * area_t) / (periment_t ** 2), 2)

        # print(f'Трикутник: площа-{area_t}, периметр-{periment_t}, центр-({cx_t}; {cy_t}), aspect_ratio-{aspect_ratio_t}, compactness-{compactness_t}')



# cv2.imshow("Mask", mask_triangle)
# cv2.imshow("figures", img)
cv2.waitKey(0)
cv2.destroyAllWindows()