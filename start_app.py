#!/usr/bin/env python3
"""
Script para iniciar la aplicación Streamlit de Pose Estimation
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Iniciar la aplicación Streamlit"""
    
    # Verificar que estamos en el directorio correcto
    if not Path("app.py").exists():
        print(" Error: No se encontró app.py en el directorio actual")
        print("   Asegúrate de ejecutar este script desde el directorio raíz del proyecto")
        sys.exit(1)
    
    # Verificar que el modelo existe
    model_path = Path("models/pytorch/pose_coco/pose_transformer_hrnet_w32_384x288.pth")
    if not model_path.exists():
        print(" Error: No se encontró el modelo preentrenado")
        print(f"   Ruta esperada: {model_path}")
        sys.exit(1)
    
    print(" Iniciando aplicación Streamlit...")
    print(" La aplicación estará disponible en: http://localhost:8501")
    print(" Presiona Ctrl+C para detener la aplicación")
    print("-" * 50)
    
    try:
        # Ejecutar Streamlit con configuraciones para evitar error 403
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.maxUploadSize", "200",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
    except KeyboardInterrupt:
        print("\n Aplicación detenida por el usuario")
    except Exception as e:
        print(f" Error al iniciar la aplicación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
