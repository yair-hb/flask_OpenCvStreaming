import cv2

captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detectRostros = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


while True:
    ret, frame = captura.read()
    if ret:
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rostros = detectRostros.detectMultiScale(gris, 1.3,5)

        for (x,y,w,h) in rostros:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('CARAS', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

captura.release()
cv2.destroyAllWindows()

