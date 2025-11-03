import cv2
net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet.caffemodel", "data/MobileNet/mobilenet_deploy.prototxt")


#2 крок - задаємо список ід та класів
classes = []
with open('data/MobileNet/synset.txt', "r", encoding = "utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(' ', 1) #Два елементи з індексами 0-id, 1-клас
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)


#3 крок - вантажимо зображення
image = cv2.imread('images/MobileNet/cat.jpg.jpg')


#4 крок - адаптуємо зображення під нашу нейронку
#blob - це зображення, яке адаптовано під нашу модель. Бере той формат, який підтримує нашу модель. Блоб - тендер
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)


#5 крок - кладемо зобраеження в мереду і запускаємо
net.setInput(blob)

preds = net.forward() #вектор йморвірності для наших класів


#6 крок - знаходимо індекс класу з найбільшою юмовірністю
idx = preds[0].argmax() #argmax - найбільше серед значень


#7 крок - дістаємо назву класу та впевненість(точність) у відсотках
label = classes[idx] if idx < len(classes) else "unknown"
conf = float(preds[0][idx]) * 100


#8 крок - виводимо результат в консоль
print("Class:", label)
print("likelihood:", conf)


#9 крок - підписуємо зображення
text = f'{label}: {int(conf)}%'
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()