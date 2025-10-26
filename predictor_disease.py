import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Nombres en espanol para las clases del dataset
DISEASE_NAMES_ES = {
    "Tomato___Bacterial_spot": "Mancha bacteriana del tomate",
    "Tomato___Early_blight": "Tizon temprano",
    "Tomato___healthy": "Tomate saludable",
    "Tomato___Late_blight": "Tizon tardio",
    "Tomato___Leaf_Mold": "Moho de la hoja",
    "Tomato___Septoria_leaf_spot": "Mancha foliar de Septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Acaro de dos manchas",
    "Tomato___Target_Spot": "Mancha diana",
    "Tomato___Tomato_mosaic_virus": "Virus mosaico del tomate",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Virus del enrollamiento amarillo del tomate",
}


class DiseasePredictor:
    def __init__(self, model_name="mymodel_v4.keras"):
        """
        Inicializa el predictor de enfermedades cargando el modelo entrenado
        desde la carpeta /models (acepta formatos .keras o .h5).
        """
        model_path = os.path.join(os.path.dirname(__file__), "models", model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ No se encontró el modelo en {model_path}")

        # 👉 Cargar modelo (Keras V3 o H5)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Modelo cargado: {model_name}")

        # ✅ Clases de tu dataset PlantVillage
        self.disease_classes = [
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___healthy",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
        ]

        # ✅ Diccionario con información adicional de cada enfermedad
        self.disease_info = {
            "Tomato___Bacterial_spot": {
                "risk_level": "Alto riesgo",
                "symptoms": "Manchas oscuras y acuosas en hojas, tallos y frutos.",
                "treatment": "Aplicar bactericidas a base de cobre, eliminar plantas infectadas.",
                "prevention": "Usar semillas certificadas, evitar riego por aspersión, rotación de cultivos."
            },
            "Tomato___Early_blight": {
                "risk_level": "Medio riesgo",
                "symptoms": "Manchas circulares con anillos concéntricos, hojas amarillentas.",
                "treatment": "Fungicidas preventivos, mejorar ventilación y reducir humedad.",
                "prevention": "Espaciamiento adecuado, evitar estrés hídrico, fertilización balanceada."
            },
            "Tomato___healthy": {
                "risk_level": "Sin riesgo",
                "symptoms": "No se detectan síntomas de enfermedad.",
                "treatment": "Mantener prácticas de manejo preventivo.",
                "prevention": "Monitoreo regular y buenas prácticas agrícolas."
            },
            "Tomato___Late_blight": {
                "risk_level": "Alto riesgo",
                "symptoms": "Manchas marrones irregulares, bordes amarillos y esporulación blanca.",
                "treatment": "Fungicidas sistémicos a base de cobre, mejorar ventilación, reducir humedad.",
                "prevention": "Rotación de cultivos, variedades resistentes, control de riego."
            },
            "Tomato___Leaf_Mold": {
                "risk_level": "Medio riesgo",
                "symptoms": "Manchas amarillas en el haz, crecimiento aterciopelado en envés.",
                "treatment": "Aplicar fungicidas específicos, mejorar circulación de aire.",
                "prevention": "Controlar humedad, espaciar plantas, reducir densidad de siembra."
            },
            "Tomato___Septoria_leaf_spot": {
                "risk_level": "Medio riesgo",
                "symptoms": "Pequeñas manchas circulares con centro gris y borde oscuro.",
                "treatment": "Eliminar hojas infectadas, aplicar fungicidas preventivos.",
                "prevention": "Evitar salpicaduras de agua, usar mulch, rotar cultivos."
            },
            "Tomato___Spider_mites Two-spotted_spider_mite": {
                "risk_level": "Alto riesgo",
                "symptoms": "Punteado amarillo, hojas bronceadas y telarañas finas.",
                "treatment": "Aplicar acaricidas específicos, aumentar humedad relativa.",
                "prevention": "Monitoreo constante, control biológico, evitar sequía."
            },
            "Tomato___Target_Spot": {
                "risk_level": "Medio riesgo",
                "symptoms": "Manchas circulares con anillos concéntricos tipo diana.",
                "treatment": "Eliminar hojas afectadas, aplicar fungicidas preventivos.",
                "prevention": "Rotación de cultivos, manejo de residuos, ventilación adecuada."
            },
            "Tomato___Tomato_mosaic_virus": {
                "risk_level": "Alto riesgo",
                "symptoms": "Patrón de mosaico verde claro y oscuro, hojas deformadas.",
                "treatment": "No hay tratamiento curativo, eliminar plantas infectadas.",
                "prevention": "Uso de semillas certificadas, desinfección de herramientas, control de vectores."
            },
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
                "risk_level": "Alto riesgo",
                "symptoms": "Amarillamiento y rizado hacia arriba de hojas, enanismo.",
                "treatment": "Eliminar plantas infectadas, control de mosca blanca.",
                "prevention": "Uso de mallas, variedades resistentes, control de vectores."
            }
        }

    def preprocess_image(self, image_file):
        """
        Preprocesa una imagen recibida para que sea compatible con MobileNetV2:
        - Convierte a RGB si no lo está
        - Redimensiona a 256x256
        - Normaliza valores entre 0 y 1
        - Añade dimensión batch
        """
        image = Image.open(io.BytesIO(image_file.read()))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((256, 256))  # Tamaño de entrada usado en entrenamiento
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    def predict(self, image_file):
        """
        Realiza la predicción sobre la imagen cargada y devuelve:
        - Enfermedad predicha
        - Confianza (%)
        - Nivel de riesgo, síntomas, tratamiento y prevención
        """
        try:
            processed_image = self.preprocess_image(image_file)
            predictions = self.model.predict(processed_image)

            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_idx]) * 100
            predicted_disease = self.disease_classes[predicted_class_idx]

            disease_data = self.disease_info.get(predicted_disease, {
                "risk_level": "Desconocido",
                "symptoms": "Información no disponible",
                "treatment": "Consultar especialista",
                "prevention": "Aplicar medidas preventivas generales"
            })

            return {
                "disease": predicted_disease,
                "disease_label": DISEASE_NAMES_ES.get(predicted_disease, predicted_disease),
                "confidence": round(confidence, 1),
                "risk_level": disease_data["risk_level"],
                "symptoms": disease_data["symptoms"],
                "treatment": disease_data["treatment"],
                "prevention": disease_data["prevention"],
            }
        except Exception as e:
            raise Exception(f"❌ Error en la predicción: {str(e)}")
