import os
import cv2
import time

from matplotlib.pyplot import imshow
from ultralytics import YOLO

from CW_less14 import CONF_THRESHOLD

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

#Для вебки або відео
USE_WEBCAM = False

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, 'name')
    cap = cv2.VideoCapture(VIDEO_PATH)


model = YOLO('yono8n') #Обрали модельку, оптимізовану для слабких комп

CONF_THRESHOLD = 0.4

RESIZE_WIDHT = 960 #розмір для відео
prev_time = time.time() #тайм
FPS = 0.0
#Описуємо цикл перегляду відео
while True:
    ret, frame = cap.read()
    if not ret: #Якщо закінчилися кадри
        break

    if RESIZE_WIDHT is not None: #Підганяємо розмір відео під задану ширину
        h, w = frame.shape[:2]
        scale = RESIZE_WIDHT / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    result = model(frame, conf = CONF_THRESHOLD, verbose = False) #Засовуємо кадри в нашу модель для отримання результату
    cat_count = 0
    dog_count = 0
    #Вказуємо "неправдиве ід" Йдуть різні люди, яким присвоюють окреме ід. якщо людина вийшла з кадру, то зазвичай її ід залишається
    #Ми виводимо не загальне ід кожної людини, а тиких які знаходяться в кадрі
    psevdo_id = 0

    CAT_CLASS_ID = 15
    DOG_CLASS_ID = 16

    for r in result: #Для кожного "результату-рамки" в резльтат
        boxes = r['boxes'] #YOLO самостійно виводить нам рамки
        if boxes is None:
            continue
        for box in boxes: #Просто стандарт
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == CAT_CLASS_ID or cls == DOG_CLASS_ID: #Якщо знайдений клас співпадає з нашим вказаним класом
                psevdo_id += 1

                if cls == CAT_CLASS_ID:
                    cat_count += 1
                    class_name = "Cat"
                else:
                    dog_count += 1
                class_name = "Dog"

                #Малюємо рамку по знайденим координатам YOLO
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f'ID f'{class_name} {psevdo_id} conf {conf:.2f}'
                #Розміщуємо текст
                cv2.putText(frame, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                now = time.time() #Кількість кадрів, які ми вже обробили
                dt = now - prev_time #Знаходимо кількість кадрів, яка пройшла від початкового моменту
                prev_time = now #перезаписуємо початок відліку

                if dt > 0: #шукаємо фпс
                    FPS = 1.0 / dt
                tota = cat_count + dog_count
                cv2.putText(frame, f'Cats: {cat_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f'Dogs: {dog_count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, f'Total: {tota}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                cv2.putText(frame, f'FPS: {fps}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                #Виводимо наше відео
                cv2,imshow("YOLO", frame)

                #Для виходу
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
cv2.destroyAllWindows()
