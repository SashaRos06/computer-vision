import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = []
y = []

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 128, 255),
    "purple": (255, 0, 255),
    "pink": (180, 105, 255),
    "white": (255, 255, 255)
}


for color_name, bgr in colors.items():
        for i in range(20):
            noise = np.random.randint(-20, 20, 3)
            sample = np.clip(np.array(bgr) + noise, 0, 255)
            X.append(sample)
            y.append(color_name)

def shape_recognition(cnt):
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
    corners = len(approx)

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = w / h
    if corners == 3:
        return "Triangle"
    elif corners == 4:
        if 0.9 < aspect_ratio < 1.2:
            return "Square"
        if 3 > aspect_ratio > 1.4:
            return "Rectangle"
    else:
        if corners > 7:
            return "Circle"
        else:
            return "Unknown"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not(ret):
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (30, 30, 30), (225, 225, 225))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            shape = shape_recognition(cnt)
            roi = frame[y:y + h, x:x + w]

            mean_color = cv2.mean(roi)[:3]
            mean_color_array = np.array(mean_color).reshape((1, -1))
            color_label = model.predict(mean_color_array)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)
            cv2.putText(frame, f"{color_label} {shape}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


    cv2.imshow("color and shape", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()