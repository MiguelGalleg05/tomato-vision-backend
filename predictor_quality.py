
import base64
import io
import os
from typing import Dict
import re

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


SPANISH_LABELS: Dict[str, str] = {
    "b_fully_ripened": "Tomate maduro",
    "fully_ripened": "Tomate maduro",
    "ripe": "Tomate maduro",
    "b_half_ripened": "Tomate en maduracion",
    "half_ripened": "Tomate en maduracion",
    "half_ripe": "Tomate en maduracion",
    "b_green": "Tomate verde",
    "green": "Tomate verde",
    "unripe": "Tomate verde",
    "b_overripe": "Tomate sobremaduro",
    "overripe": "Tomate sobremaduro",
    "damaged": "Tomate danado",
    "defect": "Tomate defectuoso",
    "premium": "Tomate premium",
}


def translate_label(value: str) -> str:
    """Traduce etiquetas del modelo a español, tolerando prefijos/sufijos.

    Maneja variantes como "Green", "b_green", "L_Green", "tomato_green".
    """
    if not isinstance(value, str) or not value.strip():
        return "Sin categoria"

    raw = value
    key = value.strip().lower()
    key = key.replace("-", "_").replace(" ", "_")
    key = key.replace("tomato", "")
    key = re.sub(r"^[a-z]_", "", key)  # elimina prefijo de una letra (b_, l_, etc.)
    key = key.strip(" _-")

    # Coincidencia directa tras normalizacion
    if key in SPANISH_LABELS:
        return SPANISH_LABELS[key]

    # Intento adicional quitando de nuevo prefijos cortos
    key2 = re.sub(r"^[a-z]_", "", key)
    if key2 in SPANISH_LABELS:
        return SPANISH_LABELS[key2]

    # Coincidencia por tokens
    token_map = {
        "fully_ripened": "Tomate maduro",
        "ripe": "Tomate maduro",
        "half_ripened": "Tomate en maduracion",
        "half_ripe": "Tomate en maduracion",
        "unripe": "Tomate verde",
        "green": "Tomate verde",
        "overripe": "Tomate sobremaduro",
        "damaged": "Tomate danado",
        "defect": "Tomate defectuoso",
        "premium": "Tomate premium",
    }
    for token, label in token_map.items():
        if token in key:
            return label

    cleaned = raw.replace("_", " ").replace("-", " ").strip()
    return cleaned.title() if cleaned else raw


class QualityPredictor:
    def __init__(self, model_path: str):
        """Inicializa el predictor de calidad/madurez con YOLOv8."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontro el modelo YOLO en {model_path}")
        self.model = YOLO(model_path)
        print(f"Modelo YOLO cargado: {model_path}")

    def _encode_annotated_image(self, base_image: Image.Image, detections: list[dict]) -> str:
        """Dibuja los cuadros y etiquetas en espa?ol y devuelve la imagen en base64."""
        annotated = base_image.copy()
        draw = ImageDraw.Draw(annotated)

        # Ajustar tama?o de fuente seg?n dimensiones de la imagen
        font_size = max(12, int(min(annotated.size) / 22))
        font: ImageFont.ImageFont
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = det["class"]
            confidence = det["confidence"]
            text = f"{label} {confidence:.0f}%"

            draw.rectangle([(x1, y1), (x2, y2)], outline="#1d4ed8", width=3)

            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except Exception:
                text_width, text_height = draw.textsize(text, font=font)
            text_x = x1 + 2
            text_y = max(0, y1 - text_height - 6)

            background = [(text_x - 3, text_y - 3), (text_x + text_width + 3, text_y + text_height + 3)]
            draw.rectangle(background, fill="#1d4ed8")
            draw.text((text_x, text_y), text, font=font, fill="#ffffff")

        buffer = io.BytesIO()
        annotated.convert("RGB").save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def predict(self, image_file):
        try:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            results = self.model.predict(image, imgsz=640, conf=0.5)

            detections: list[dict] = []
            class_counts: Dict[str, int] = {}

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model.names[cls_id]
                    translated = translate_label(class_name)

                    bbox = [round(float(coord), 2) for coord in box.xyxy[0].tolist()]

                    detections.append(
                        {
                            "class": translated,
                            "raw_class": class_name,
                            "confidence": round(conf * 100, 2),
                            "bbox": bbox,
                        }
                    )

                    key = class_name.lower()
                    class_counts[key] = class_counts.get(key, 0) + 1

            annotated_image = self._encode_annotated_image(image, detections)

            return {
                "detections": detections,
                "total": len(detections),
                "class_counts": class_counts,
                "annotated_image": annotated_image,
            }
        except Exception as exc:
            raise Exception(f"Error en prediccion YOLO: {exc}")
