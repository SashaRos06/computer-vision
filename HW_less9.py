import cv2
net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')

classes = []
with open('data/MobileNet/synset.txt', "r", encoding = "utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)



image, image1, image2, image3, image4 = cv2.imread('images/MobileNet/cat.jpg'), cv2.imread('images/MobileNet/dog.jpg'), cv2.imread('images/MobileNet/macaw.jpg'), cv2.imread('images/MobileNet/hamster.jpg'), cv2.imread('images/MobileNet/scorpion.jpg')

blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)
blob1 = cv2.dnn.blobFromImage(
    cv2.resize(image1, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)
blob2 = cv2.dnn.blobFromImage(
    cv2.resize(image2, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)
blob3 = cv2.dnn.blobFromImage(
    cv2.resize(image3, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)
blob4 = cv2.dnn.blobFromImage(
    cv2.resize(image4, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)


net.setInput(blob)
preds = net.forward()
idx = preds[0].argmax()

net.setInput(blob1)
preds1 = net.forward()
idx1 = preds1[0].argmax()

net.setInput(blob2)
preds2 = net.forward()
idx2 = preds2[0].argmax()

net.setInput(blob3)
preds3 = net.forward()
idx3 = preds3[0].argmax()

net.setInput(blob4)
preds4 = net.forward()
idx4 = preds4[0].argmax()

label = classes[idx] if idx < len(classes) else "unknown"
label1 = classes[idx1] if idx1 < len(classes) else "unknown"
label2 = classes[idx2] if idx2 < len(classes) else "unknown"
label3 = classes[idx3] if idx3 < len(classes) else "unknown"
label4 = classes[idx4] if idx4 < len(classes) else "unknown"

conf = float(preds[0][idx]) * 100
conf1 = float(preds1[0][idx1]) * 100
conf2 = float(preds2[0][idx2]) * 100
conf3 = float(preds3[0][idx3]) * 100
conf4 = float(preds4[0][idx4]) * 100

print("Class:", label)
print("likelihood:", conf)

print("Class:", label1)
print("likelihood:", conf1)

print("Class:", label2)
print("likelihood:", conf2)

print("Class:", label3)
print("likelihood:", conf3)
print("Class:", label4)
print("likelihood:", conf4)

text = f'{label}: {int(conf)}%'
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

text1 = f'{label1}: {int(conf1)}%'
cv2.putText(image1, text1, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

text2 = f'{label2}: {int(conf2)}%'
cv2.putText(image2, text2, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

text3 = f'{label3}: {int(conf3)}%'
cv2.putText(image3, text3, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

text4 = f'{label4}: {int(conf4)}%'
cv2.putText(image4, text4, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

cv2.imshow("result", image)
cv2.imshow("result1", image1)
cv2.imshow("result2", image2)
cv2.imshow("result3", image3)
cv2.imshow("result4", image4)
cv2.waitKey(0)
cv2.destroyAllWindows()