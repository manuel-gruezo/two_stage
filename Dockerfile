# ===========================================
# BASE IMAGE: Python para ARM64 (compatible con Apple Silicon)
# ===========================================
FROM python:3.10-slim

# Evitar prompts interactivos y mejorar pip performance
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Crear y establecer directorio de trabajo
WORKDIR /app

# ===========================================
# DEPENDENCIAS DEL SISTEMA
# ===========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    g++ \
    make \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ===========================================
# DEPENDENCIAS PYTHON
# ===========================================
# Copiar requirements primero (mejor cacheo)
COPY requirements.txt /app/requirements.txt

# Actualizar pip y herramientas básicas
RUN pip install --upgrade pip setuptools wheel

# Instalar dependencias (PyTorch se instalará desde requirements.txt con soporte para ARM64)
RUN pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

# ===========================================
# COPIAR CÓDIGO DE LA APLICACIÓN
# ===========================================
COPY . /app

# Crear carpetas necesarias con permisos adecuados
RUN mkdir -p /app/input /app/output /app/models/pytorch/pose_coco /app/.streamlit

# ===========================================
# USUARIO SEGURO (no root)
# ===========================================
RUN useradd -m -s /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# ===========================================
# PUERTO Y COMANDO DE ARRANQUE
# ===========================================
EXPOSE 8501

# Si tu app es Streamlit
CMD ["streamlit", "run", "start_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

