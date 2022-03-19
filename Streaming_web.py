from flask import Flask
from flask import render_template
from flask import Response
import cv2

app = Flask(__name__)
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detectRostros = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

def generate():
    while True:
        ret, frame = captura.read()
        if ret:
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rostros = detectRostros.detectMultiScale(gris, 1.3,5)

            for (x,y,w,h) in rostros:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n'+ bytearray(encodedImage)+ b'\r\n')

@app.route("/")
def index ():
    return render_template('index.html')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype= "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False)
    
captura.release()
