#Обробка відео
import cv2
#Цикл, який буде перебирати кожне зображення та обробляти його

cap = cv2.VideoCapture(0) #0-працює з вебкамерами. Також можна вказати шлях до нашого відео

ret, frame1 = cap.read()

grey1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
grey1 = cv2.convertScaleAbs(grey1, alpha=1.5, beta=15)  #Контраст 1-коефіцієнт контрастності, 2-зміна яскравості



while True:
    ret, frame2 = cap.read() #frame - кадр, який перебирається. ret - чи відкритий кадр True або False
    if not ret: #Якщо не отримали кадри
        print("Відео скінчилося")
        break

    grey2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.convertScaleAbs(grey2, alpha=1.5, beta=15)

    diff = cv2.absdiff(grey1, grey2) #Знаходимо різницю
    _, thesh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thesh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Video', frame2) #Виводимо наші кадри


    if cv2.waitKey(1) and 0xFF == ord('q'):
        break


cap.release() #Коли ми будемо запускати програму поток відеокамери буде зайнятий. Для того, зоб кожного разу ми переставали працювати з відеокартоб. Ми вказуємо, коли закриваємо вебку ми віключаємося від неї
cv2.destroyAllWindows()