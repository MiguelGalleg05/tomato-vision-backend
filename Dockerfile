FROM python:3.11-slim

# Variables de entorno básicas
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Establece el directorio de trabajo correcto
WORKDIR /app/python_backend

# Copia los archivos del backend
COPY ./python_backend /app/python_backend

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto
EXPOSE 5000

# Comando para ejecutar Flask
CMD ["python", "app.py"]
