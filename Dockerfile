FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app/python_backend

# Copia los archivos desde la carpeta correcta
COPY ./Trabajo_Grado/python_backend /app/python_backend

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
