import traceback
try:
    import app.detection.yolov5_detector as m
    print("✅ Módulo importado OK:", m.__file__)
    print("📦 Clases en el módulo:", [x for x in dir(m) if x.endswith("Detector")])
    print("🧩 Tiene OnnxYoloV5Detector:", hasattr(m, "OnnxYoloV5Detector"))
except Exception as e:
    print("❌ Fallo al importar:")
    print(e)
    traceback.print_exc()
