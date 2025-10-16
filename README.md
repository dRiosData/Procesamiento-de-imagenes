Detector de Seguridad - MVP

Proyecto MVP de detección de objetos basado en YOLOv5 (formato ONNX) e implementado con FastAPI.

El sistema permite cargar imágenes y detectar objetos en ellas, devolviendo tanto los resultados en formato JSON como la imagen procesada con los cuadros delimitadores y etiquetas.

⚙️ Requisitos previos

Python 3.10

pip actualizado

Modelo ONNX: models/yolov5s.onnx (ya incluido en el proyecto)


🧩 Instalación del entorno

Clonar el repositorio o copiar el proyecto:

git clone https://github.com/usuario/tp-detector-seguridad.git
cd tp-detector-seguridad


Crear un entorno virtual:

python -m venv .venv


Activar el entorno virtual:

# En PowerShell (Windows)
.\.venv\Scripts\Activate


Instalar las dependencias:

pip install -r requirements.txt

🚀 Ejecución del servidor

Ejecutar el servidor local con:

python -m uvicorn app.api.main:app --reload


Verás un mensaje similar a:

Uvicorn running on http://127.0.0.1:8000
Application startup complete.

🌐 Uso desde el navegador

Abrir la documentación interactiva de FastAPI:

👉 http://127.0.0.1:8000/docs

Allí podrás probar los endpoints desde la interfaz Swagger.


📡 Endpoints principales
1. **GET /health

Verifica el estado del servidor.

Ejemplo de respuesta:

{"status": "ok"}

2. POST /api/v1/detect-image

Recibe una imagen y devuelve detecciones en formato JSON.

Parámetros:

file: imagen a analizar

conf_threshold: (opcional) umbral de confianza (default: 0.25)

Ejemplo de respuesta:

{
  "filename": "ejemplo.jpg",
  "count": 2,
  "detections": [
    {
      "box": [80, 100, 200, 240],
      "score": 0.87,
      "cls": 0
    },
    {
      "box": [320, 50, 460, 200],
      "score": 0.76,
      "cls": 16
    }
  ],
  "params": {"conf_threshold": 0.25}
}


3. POST /api/v1/detect-image-viz

Devuelve la imagen procesada con los cuadros y etiquetas visuales (formato JPEG).

Ejemplo visual:

Entrada: foto de un perro.

Salida: imagen con un cuadro verde y etiqueta "dog 0.73".


📁 Estructura del proyecto
tp-detector-seguridad/
│
├── app/
│   ├── api/
│   │   ├── main.py
│   ├── detection/
│   │   ├── yolov5_detector.py
│   │   ├── base.py
│   └── __init__.py
│
├── models/
│   └── yolov5s.onnx
│
├── notebooks/
├── .venv/
├── requirements.txt
├── test_import.py
└── README.md

🧾 Créditos y licencia

Proyecto desarrollado como MVP académico para el curso de Inteligencia Artificial para Videovigilancia.
Basado en el modelo preentrenado YOLOv5s (Ultralytics).