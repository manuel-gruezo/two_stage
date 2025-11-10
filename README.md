# PRTR Two-stage Cascade Transformers

## Información del Artículo Base

**Nombre del artículo:** Pose Recognition with Cascade Transformers

**Enlace al artículo:** [Pose Recognition with Cascade Transformers](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Pose_Recognition_With_Cascade_Transformers_CVPR_2021_paper.pdf)

**Repositorio original:** [PRTR en GitHub](https://github.com/mlpc-ucsd/PRTR/tree/main)

## Descripción 

Esta aplicación ofrece una interfaz web (Streamlit) para realizar detección de poses humanas en imágenes utilizando el modelo PRTR (Pose Recognition with Cascade Transformers). Esta técnica detecta personas y estima sus 17 keypoints corporales mediante una arquitectura de Transformers en cascada, permitiendo identificar y analizar poses con alta precisión y eficiencia. A diferencia de los enfoques tradicionales basados en heatmaps, PRTR adopta un método de regresión directa que elimina la necesidad de complejos pre-procesamientos y post-procesamientos heurísticos.


## Interfaz



## Ejemplos






### Innovaciones principales:

**Arquitectura Two-Stage en Cascada:**
- **Person-Detection Transformer (DETR)**: Detecta personas en la imagen completa usando el mecanismo de matching bipartito de DETR. Aunque DETR puede generar detecciones duplicadas, se aplica NMS (Non-Maximum Suppression) para eliminar duplicados y evitar detectar la misma persona múltiples veces.
- **Keypoint-Detection Transformer (PRTR)**: Predice los 17 keypoints COCO para cada persona detectada mediante regresión directa de coordenadas, haciendo el pipeline completamente diferenciable. Este modelo NO requiere NMS ya que usa Hungarian Matching para asignar queries a keypoints.
- **Mecanismo de Queries**: 100 queries aprendidas que compiten para predecir los 17 keypoints, asignadas óptimamente mediante Hungarian Matching basado en probabilidad de clase.
- **Refinamiento Gradual**: Las queries de keypoints se refinan progresivamente a través de las capas del decoder, mejorando iterativamente las predicciones.

## Resumen Teórico de la Arquitectura

### Arquitectura Two-Stage

![model_two_stage](https://raw.githubusercontent.com/mlpc-ucsd/PRTR/refs/heads/main/figures/model_two_stage.png)

La arquitectura two-stage de PRTR consiste en:

#### 1. Person-Detection Transformer (DETR)

- **Backbone CNN**: Extrae características de la imagen completa
- **Transformer Encoder-Decoder**: Procesa las características con atención
- **Salida**: Detecta personas con bounding boxes usando el mecanismo de matching bipartito de DETR
- **NMS**: Aunque DETR usa matching bipartito, puede generar detecciones duplicadas. Por lo tanto, se aplica NMS (Non-Maximum Suppression) para eliminar duplicados y evitar detectar la misma persona múltiples veces.

#### 2. Keypoint-Detection Transformer (PRTR)

- **Input**: Recortes de personas detectadas (expandidos 25% para contexto adicional)
- **Procesamiento**: Cada recorte se procesa independientemente con transformación afín al tamaño del modelo (512x384 píxeles, width x height)
- **Mecanismo de Queries**: 100 queries aprendidas que compiten para predecir los 17 keypoints
- **Hungarian Matching**: Asigna óptimamente queries a keypoints usando el algoritmo de asignación lineal (linear_sum_assignment) basado en las probabilidades de clase. Este matching se realiza internamente en `get_final_preds_match()`.
- **Filtrado Posterior**: Después del Hungarian Matching, se aplican filtros adicionales para validar keypoints:
  - Probabilidad de clase > 30%
  - Distancia espacial < 50 píxeles
  - Confianza del heatmap > 80%
- **Salida**: 17 keypoints en coordenadas absolutas con sus confianzas

### Proceso de Inferencia

1. **Detección**: El Person-Detection Transformer identifica todas las personas en la imagen
2. **Recorte**: Cada detección se expande un 25% y se recorta
3. **Normalización**: Transformación afín al tamaño del modelo (512x384 píxeles, width x height) usando centro y escala calculados del recorte
4. **Predicción**: Keypoint-Detection Transformer procesa cada recorte
5. **Flip-Test**: Promedio de predicciones con imagen espejo para mayor robustez
6. **Transformación Inversa**: Coordenadas se mapean de vuelta al espacio original:
   - Espacio del modelo → Espacio del recorte (transformación afín inversa)
   - Espacio del recorte → Imagen original (aplicación de offsets de bounding box)

## Pasos para ejecutar el proyecto

### Prerrequisitos

- Python 3.9+
- CUDA (opcional, para aceleración GPU)
- 8GB+ RAM recomendado

### 1) Ejecución Local

#### Configuración del Entorno

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate

# Activar entorno (Linux/Mac)
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Estructura de Modelos

Asegurar la siguiente estructura de archivos:

```
models/
├── pytorch/
│   └── pose_coco/
│       └── pose_transformer_hrnet_w32_512x384.pth
└── detr-resnet-101/
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```
Descargar en el siguinte link [link models](https://drive.google.com/drive/folders/1eJCNwf9BnJf0YoOLLk5rKQw7pdFtb33w?usp=sharing)

#### Ejecución por Lotes

El script `testInference.py` procesa todas las imágenes del directorio `local/input/` y guarda los resultados en `local/output/`.

```bash
# Procesar todas las imágenes en local/input/
python local/testInference.py
```

**Funcionamiento:**
- Lee automáticamente todas las imágenes del directorio `local/input/` (formatos: `.jpg`, `.jpeg`, `.png`, `.bmp`)
- Aplica la misma lógica de inferencia que `app.py` (DETR + NMS + PRTR)
- Procesa cada imagen y detecta personas con sus keypoints
- Guarda los resultados en `local/output/`:
  - **Imágenes**: `{nombre_imagen}_pose.jpg` - Imágenes con poses detectadas dibujadas
  - **JSON**: `{nombre_imagen}_detections.json` - Metadatos detallados con coordenadas, confianzas y bounding boxes por persona

**Parámetros de configuración** (definidos en el script):
- `DETR_THRESHOLD = 0.9` - Umbral de confianza para detección DETR
- `NMS_THRESHOLD = 0.5` - Umbral IoU para eliminar detecciones duplicadas
- `POINT_SIZE_MULTIPLIER = 1.0` - Tamaño de los puntos de keypoints
- `LINE_WIDTH_MULTIPLIER = 1.0` - Grosor de las líneas del esqueleto

### 2) Despliegue con Docker

#### Construcción y Ejecución

```bash
# Construir imagen
docker compose build

# Ejecutar contenedor
docker compose up
```

#### Acceso a la Aplicación

- Abrir en el navegador: `http://localhost:8501`
- Para detener el servicio: `Ctrl + C`

### 3) Uso de la Aplicación Streamlit

#### Modo "Análisis de Imagen"
- Subir una imagen (formatos: JPG, JPEG, PNG, BMP)
- Ajustar parámetros en la barra lateral:
  - **Umbral DETR**: Controla sensibilidad de detección (0.9 recomendado)
  - **Umbral NMS**: Elimina duplicados (0.5 recomendado)
  - **Tamaño puntos**: Ajusta visualización de keypoints
  - **Grosor líneas**: Controla esqueleto visible
- Presionar "Analizar Pose"

#### Modo "Tomar Foto"
- Permitir acceso a cámara
- Capturar foto
- Presionar "Analizar Pose"

#### Resultados
- Imagen con poses superpuestas
- Tabla de keypoints por persona
- Coordenadas y confianzas detalladas
- Opción de descarga de resultados

## ¿Cómo se cargan los pesos?

La función `load_models()` en `app.py`:

1. **Modelo de Pose (PRTR)**:
   - Lee la configuración desde `experiments/coco/transformer/w32_512x384_adamw_lr1e-4.yaml`
   - Carga la arquitectura PRTR-HRNet W32
   - Inyecta los pesos preentrenados desde `models/pytorch/pose_coco/pose_transformer_hrnet_w32_512x384.pth`
   - Pone el modelo en modo evaluación (`eval()`)

2. **Modelo DETR (Detección)**:
   - Carga el procesador y modelo DETR desde la carpeta local `models/detr-resnet-101/`
   - Usa `DetrImageProcessor` para preprocesamiento
   - Usa `DetrForObjectDetection` para inferencia

3. **Dispositivo**:
   - Selecciona automáticamente `cuda` si está disponible, en caso contrario `cpu`

**Rutas importantes (por defecto en `app.py`)**:
- Pose: `models/pytorch/pose_coco/pose_transformer_hrnet_w32_512x384.pth`
- DETR: `models/detr-resnet-101/` (directorio con los ficheros del modelo)

**Nota**: Si cambias la ubicación de los modelos, ajusta en `app.py`:
- Ruta del pose: atributo `pretrained` de la clase `Args`
- Ruta del DETR: variable `local_model_path` dentro de `load_models()`

## ¿Cómo se realiza la inferencia?

### Pipeline de Inferencia

#### Fase 1: Detección de Personas

1. **Procesamiento DETR**:
   - Imagen completa → características extraídas
   - Post-procesamiento: filtrado por clase "persona" (label == 1 en COCO)
   - Umbral de confianza configurable (0.9 recomendado)

2. **NMS (Non-Maximum Suppression)**:
   - **Nota importante**: Aunque DETR usa matching bipartito, puede generar detecciones duplicadas de la misma persona
   - Se aplica NMS para eliminar estas detecciones duplicadas
   - Cálculo de IoU (Intersection over Union) entre cajas
   - Umbral IoU configurable (0.5 recomendado)
   - Esto asegura que cada persona sea detectada solo una vez

3. **Expansión de Bounding Boxes**:
   - Cada caja se expande 25% en cada dirección (50% total)
   - Proporciona contexto adicional para keypoints cerca de los bordes
   - Mejora la detección de extremidades

#### Fase 2: Estimación de Pose

1. **Transformación Afín**:
   - **Centro**: `(w/2, h/2)` del recorte
   - **Escala**: `max(w, h) / 200.0 * 1.25` (factor 1.25 para margen)
   - **Tamaño objetivo**: 512x384 píxeles (width x height)

2. **Inferencia del Modelo**:
   - **Forward pass original**: Recorte transformado → predicción
   - **Forward pass flipped**: Imagen espejo → predicción
   - **Promedio**: `(predicción_original + predicción_flipped) / 2.0`
   - Mayor robustez ante variaciones de orientación

3. **Hungarian Matching**:
   - Se ejecuta internamente en `get_final_preds_match()` usando `linear_sum_assignment`
   - 100 queries aprendidas compiten para predecir 17 keypoints
   - Asignación óptima basada en las probabilidades de clase (matriz de costos negativa)
   - El algoritmo encuentra la mejor asignación 1-a-1 entre queries y keypoints

4. **Filtrado Posterior**:
   - Después del Hungarian Matching, se aplican filtros adicionales para validar keypoints:
     - **Probabilidad de clase** > 30%
     - **Distancia espacial** < 50 píxeles
     - **Confianza del heatmap** > 80%
   - Si no se cumplen condiciones estrictas, se buscan queries de soporte cercanas
   - Keypoints con confianza > 0.8 se aceptan
   - Keypoints con baja confianza se enmascaran como NaN

5. **Transformación Inversa**:
   - **Paso 1**: Coordenadas modelo → espacio recorte (transformación afín inversa)
   - **Paso 2**: Espacio recorte → imagen original (aplicación de offsets de bounding box)
   - Garantiza que los keypoints se dibujen en la posición correcta

#### Fase 3: Visualización

1. **Dibujado del Esqueleto**:
   - Conexión de pares de joints según definición COCO
   - Líneas coloridas por segmento corporal:
     - Verde: brazo izquierdo
     - Amarillo: brazo derecho
     - Azul: pierna izquierda
     - Rosa: pierna derecha
     - Rosa claro: conexiones de cabeza

2. **Tablas de Resultados**:
   - Detalle por persona con coordenadas (x, y) y confianza para cada keypoint
   - Información de bounding box usado para el recorte

### Configuraciones Clave

#### Parámetros DETR
- **Umbral confianza**: 0.9 (balance precisión/recall)
- **NMS threshold**: 0.5 (eliminación duplicados)
- **Expansión bbox**: 25% (mejor contexto para keypoints)

#### Parámetros PRTR
- **Tamaño entrada**: 512x384 píxeles (width x height, óptimo para HRNet-W32)
- **Flip-test**: Habilitado (mejora robustez)
- **Umbral keypoints**: 0.8 (filtrado conservador)
- **Hungarian Matching**: Automático (basado en probabilidades de clase)
- **Filtrado posterior**: Probabilidad > 30%, distancia < 50px, confianza > 80%

## Agradecimientos
Este proyecto está basado en los siguientes repositorios de código abierto, que facilitan enormemente nuestra investigación.

- Gracias a [DETR](https://github.com/facebookresearch/detr) por la implementación de [Detection Transformer](https://arxiv.org/abs/2005.12872)

- Gracias a [PRTR](https://github.com/mlpc-ucsd/PRTR/tree/main) por la implementación de [Pose Recognition with Cascade Transformers](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Pose_Recognition_With_Cascade_Transformers_CVPR_2021_paper.pdf)


## Herramientas Utilizadas

- 🧰 **[Git](https://git-scm.com/)**
- 🗂 **[GitHub](https://github.com/)**
- 🐍 **[Python](https://www.python.org/)**
- 🔥 **[Pytorch](https://pytorch.org/)**
- 🐳 **[Docker](https://www.docker.com/)**
- 📈 **[Streamlit](https://streamlit.io/)**
- 🧬 **[Hugging Face Transformers](https://huggingface.co/docs/transformers)**



## Autores

- **Jhonatan Steven Morales / jhonatan19991**  
- **Manuel Alejandro Perlaza / manuel-gruezo**  
- **Carol Dayana Varela / caroldvarela**  