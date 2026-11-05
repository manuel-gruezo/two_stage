#!/bin/bash

# Script para subir el proyecto a GitHub
# Reemplaza 'TU_NOMBRE_REPOSITORIO' con el nombre que le diste en GitHub

# Ejemplo: si tu repositorio se llama "two_stage", la URL sería:
# https://github.com/manuel-gruezo/two_stage.git

echo "Por favor, ingresa el nombre de tu repositorio en GitHub:"
read REPO_NAME

# Conectar el repositorio local con GitHub
git remote add origin https://github.com/manuel-gruezo/$REPO_NAME.git

# Verificar que se agregó correctamente
echo "Verificando conexión..."
git remote -v

# Subir el código a GitHub
echo "Subiendo código a GitHub..."
git branch -M main
git push -u origin main

echo "¡Listo! Tu código debería estar en GitHub ahora."
echo "Visita: https://github.com/manuel-gruezo/$REPO_NAME"

