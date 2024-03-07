import cv2
import numpy as np

# Cargar el clasificador de detección de rostros preentrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Obtener el elemento canvas y video del DOM
canvas = document.querySelector('#canvas')
video = document.querySelector('#video')

# Configurar la función de actualización de video
def update_video():
    global canvas, video, face_cascade, cap
    ret, frame = cap.read()
    if not ret:
        return

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar un rectángulo alrededor de cada rostro detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convertir el frame a una matriz numpy de 32-bit float
    frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255

    # Convertir la matriz numpy a un objeto ImageData
    img_data = document.createImageDataFromArray(frame_np)

    # Dibujar el frame procesado en el canvas
    canvas.getContext('2d').putImageData(img_data, 0, 0)

    # Llamar a la función de actualización de video cada 33 ms (30 fps)
    document.py_schedule(update_video, 33)

# Iniciar la función de actualización de video
update_video()
