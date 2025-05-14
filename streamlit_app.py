import streamlit as st
import cv2
import math
from ultralytics import YOLO
import numpy as np

st.set_page_config(page_title="YOLOv8 Animal Detection", layout="wide")

st.title("🦁 Detección de animales con YOLOv8")
st.markdown("Este demo usa la webcam para detectar animales usando un modelo YOLOv8 preentrenado.")

# Carga del modelo
@st.cache_resource
def load_model():
    return YOLO("yolo-Weights/yolov8n.pt")

model = load_model()

# Clases permitidas
allowed_classes = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]

# Botón para activar cámara
start = st.button("📷 Iniciar cámara")

frame_window = st.image([])

if start:
    captura = cv2.VideoCapture(0)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    st.info("Presiona 'Detener' para finalizar la cámara.")
    stop = st.button("⛔ Detener")

    while captura.isOpened():
        success, img = captura.read()
        if not success:
            st.error("No se puede acceder a la cámara.")
            break

        # Realiza detección
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                class_name = model.names[cls]

                if class_name in allowed_classes:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img, f'{class_name} {confidence}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)

        # Convertir BGR a RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_window.image(img_rgb)

        # Permitir detener el proceso
        if stop:
            captura.release()
            break
