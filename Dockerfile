FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# 🔧 Instalar dependencias del sistema necesarias para OpenCV y YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 📦 Instalar dependencias Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 📁 Copiar todo el proyecto
COPY . /app

# 🌐 Exponer puerto Flask
EXPOSE 5000

# 🚀 Ejecutar la app
CMD ["python", "app.py"]
