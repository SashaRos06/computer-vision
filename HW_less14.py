import cv2
import numpy as np
import os
import shutil


PROJECT_DIR = os.path.dirname(__file__) #створюємо наш основний файл

IMAGES_DIR = os.path.join(PROJECT_DIR, "images") #в загальній пацпці тека для зображень
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output") #папка для загального виводу зображень
PEOPLE_DIR = os.path.join(OUTPUT_DIR, "people")
NO_PEOPLE_DIR = os.path.join(OUTPUT_DIR, "no_people")

os.makedirs(PEOPLE_DIR, exist_ok = True)
os.makedirs(NO_PEOPLE_DIR, exist_ok = True)

PROTOTXT_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")

net = cv2.dnn.readNet(PROTOTXT_PATH, MODEL_PATH)



CLASSES = ["background","aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair","cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant","sheep", "sofa", "train", "tvmonitor"]
PERSON_CLASS_ID = CLASSES.index("person")
#Прописуємо поріг впевненості
CONF_THRESHOLD = 0.6

def detect_person(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor = 0.007843, size = (300, 300), mean = (127.5, 127.5, 127.5)) #blob - формат під нейронку
    net.setInput(blob) #запускаємо мережу в роботу
    detection = net.forward() #видає прогноз точності

    people = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        class_id = detection[0, 0, i, 1]

        if class_id == PERSON_CLASS_ID and confidence >= CONF_THRESHOLD:
            box = detection[0, 0, i, 3:7]
            #Переводимо координати в рамку для детекції найб. confidence
            x1 = int(box[0]*w)
            y1 = int(box[1]*h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            people.append((x1, y1, x2, y2, confidence))
    return people

#Вказуємо дозволені розширення, з якими працює мережа
allowed_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
files = os.listdir(IMAGES_DIR)

count_people = 0
count_no_people = 0

for file in files:
    if not file.lower().endswith(allowed_ext):
        continue

    in_path = os.path.join(IMAGES_DIR, file) #розташування нашого файлу
    img = cv2.imread(in_path)
    people = detect_person(img)
    N = len(people)


    if N > 0:
        count_people += 1

        boxed = img.copy()
        for (x1, y1, x2, y2, conf) in people:
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(boxed, f'person: {conf:.2f}', (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.putText(boxed, f"People on photo: {N}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        boxed_path = os.path.join(PEOPLE_DIR, "boxed_" + file)
        cv2.imwrite(boxed_path, boxed)

else:
    count_no_people += 1
    out_path = os.path.join(NO_PEOPLE_DIR, file)
    shutil.copyfile(in_path, out_path)


print("Фото з людьми:", count_people)
print("Фото без людей:", count_no_people)