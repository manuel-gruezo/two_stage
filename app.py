import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
import sys
import os
import io
import cv2
import time
import tempfile
import hashlib
from transformers import DetrImageProcessor, DetrForObjectDetection
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque


# Configurar la página
st.set_page_config(
    page_title="Pose Estimation App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        color: #333333;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .camera-container {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        background: #f8f9ff;
        margin: 1rem 0;
        color: #333333;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .button-primary {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9ff 0%, #e8f0ff 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    .stSlider > div > div > div {
        background-color: #667eea;
    }
    
    .stSlider > div > div > div > div {
        background-color: #667eea;
    }
    
    .slider-container {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Añadir el directorio lib al path para imports
sys.path.insert(0, '/app/lib')


import pillow_avif
# Imports del proyecto
from config import cfg as conf
from config import update_config
from utils.utils import model_key_helper
from core.inference import get_final_preds_match
from utils.transforms import get_affine_transform
import models

# ---------- Configuración del modelo ----------
class Args:
    cfg = 'experiments/coco/transformer/w32_512x384_adamw_lr1e-4.yaml'
    opts = []
    modelDir = None
    logDir = None
    dataDir = None
    pretrained = 'models/pytorch/pose_coco/pose_transformer_hrnet_w32_512x384.pth'

@st.cache_resource
def load_models():
    """
    Carga y prepara los modelos necesarios para la app.

    NOTA SOBRE CACHING: Esta función usa @st.cache_resource de Streamlit para cachear los modelos.
    Esto evita recargarlos en cada interacción del usuario, mejorando significativamente la velocidad.
    Los modelos se cargan una sola vez y se reutilizan en todas las inferencias.
    
    Returns:
        tuple: (pose_model, detr_model, detr_processor, device, conf) o (None, None, None, None, None) si hay error
    
    Nota: Si hay error, se muestra un mensaje de error en Streamlit y se retorna None para cada componente.
    """

    args = Args()
    update_config(conf, args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar modelo de pose
    pose_model = models.pose_transformer.get_pose_net(conf, is_train=False)
    state = torch.load(args.pretrained, map_location='cpu')
    pose_model.load_state_dict(model_key_helper(state), strict=False)
    pose_model.to(device)
    pose_model.eval()
    
    # Cargar modelo DETR
    local_model_path = "models/detr-resnet-101"
    detr_processor = DetrImageProcessor.from_pretrained(local_model_path)
    detr_model = DetrForObjectDetection.from_pretrained(local_model_path)
    
    return pose_model, detr_model, detr_processor, device, conf

# ---------- Normalización ----------
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Tamaño del modelo de pose (se usa en get_pose_keypoints)
MODEL_W, MODEL_H = 384, 512

# ---------- Esqueleto para dibujar - COCO ----------
# Skeleton en formato 0-based (índices de array Python)
SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4], [3, 5], [4, 6]]

# ---------- Funciones auxiliares para metadatos y preprocesamiento ----------
def make_meta_from_wh(w, h, use_max_scale=True):
    """
    Calcula metadatos de centro y escala para transformación afín.
    
    Args:
        w (int): Ancho del recorte en píxeles
        h (int): Alto del recorte en píxeles  
        use_max_scale (bool): Si True, usa la dimensión máxima (max(w,h)) para calcular escala
    
    Returns:
        tuple: (center, scale) donde:
            center (np.array): [x_center, y_center] en píxeles del recorte
            scale (np.array): [scale_x, scale_y] factores de escala para la transformación afín
    
    Nota: La escala incluye un factor de 1.25 para añadir margen alrededor de la persona,
    mejorando la detección de keypoints cerca de los bordes del recorte.
    """
    if use_max_scale:
        s_val = max(w, h) / 200.0
        scale = np.array([s_val, s_val], dtype=np.float32) * 1.25
    else:
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32) * 1.25
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    return center, scale

def preprocess_patch(img_pil, center, scale, output_size):
    """
    Aplica una transformación afín para llevar el recorte al tamaño del modelo.

    Args:
        img_pil (PIL.Image): Imagen del recorte de la persona
        center (np.array): [x_center, y_center] del recorte
        scale (np.array): [scale_x, scale_y] factores de escala
        output_size (tuple): (width, height) tamaño esperado por el modelo (ej: 512x384)
    
    Returns:
        np.array: Imagen transformada y redimensionada al tamaño del modelo
    
    Nota: Usa `center` y `scale` para construir la matriz afín y re-muestrear la imagen
    con interpolación bilineal, conservando las proporciones esperadas por el modelo.
    """
    trans = get_affine_transform(center, scale, 0, output_size)
    img_np = np.array(img_pil)
    out = cv2.warpAffine(img_np, trans, (int(output_size[0]), int(output_size[1])), flags=cv2.INTER_LINEAR)
    return out

# ---------- Dibujo de keypoints y esqueleto ----------
"""
DISEÑO VISUAL:

Esqueleto con líneas coloridas por segmento corporal:
   Verde: brazo izquierdo (hombro → codo → muñeca)
   Amarillo: brazo derecho (hombro → codo → muñeca)
   Azul: pierna izquierda (cadera → rodilla → tobillo)
   Rosa: pierna derecha (cadera → rodilla → tobillo)
   Rosa claro: conexiones de cabeza (líneas delgadas)
   
Visualización con círculos negros para los puntos clave
"""
def draw_keypoints_pil(img_pil, keypoints, confidences=None, scale=1.0, 
                      point_size_multiplier=1.0, line_width_multiplier=1.0):
    """
    Dibuja keypoints y conexiones del esqueleto sobre una imagen PIL.

    Args:
        img_pil (PIL.Image): Imagen sobre la cual dibujar (se modifica in-place)
        keypoints (np.array): Array de shape (17, 2) con coordenadas [x, y] de cada keypoint
        confidences (np.array, optional): Array de shape (17,) con confianza de cada keypoint
        scale (float): Factor de escala para ajustar tamaño de puntos y líneas
        point_size_multiplier (float): Multiplicador adicional para tamaño de puntos
        line_width_multiplier (float): Multiplicador adicional para grosor de líneas
    
    Returns:
        PIL.Image: Imagen modificada con keypoints y esqueleto dibujados
    
    Nota: Los keypoints ya fueron filtrados en get_pose_keypoints (conf_th = 0.8).
    Solo se dibujan los keypoints con score > 0 (los filtrados tienen score = 0.0).
    Los keypoints con coordenadas NaN no se visualizan.
    """
    draw = ImageDraw.Draw(img_pil)
    
    # Colores EXACTOS de la clase Visualizer (0-based) - COMPLETOS
    GREEN = [(4,5),(5,7),(7,9)]           # left_shoulder->left_elbow->left_wrist
    YELLOW = [(4,6),(6,8),(8,10)]         # right_shoulder->right_elbow->right_wrist
    BLUE = [(5,11),(11,13),(13,15)]       # left_shoulder->left_hip->left_knee->left_ankle
    PINK = [(6,12),(12,14),(14,16)]       # right_shoulder->right_hip->right_knee->right_ankle
    
    # Convertir confidences a vis mask
    # Los keypoints ya fueron filtrados en get_pose_keypoints (score > 0.8)
    # Los filtrados tienen score = 0.0, así que solo verificamos score > 0
    vis = np.ones((17,), dtype=np.int32)  # Por defecto todos visibles
    if confidences is not None:
        confidences = np.asarray(confidences)
        vis = (confidences > 0.0).astype(np.int32)  # Solo verificar que no sea 0 (ya filtrado)
    
    # Dibujar líneas del skeleton
    for i_idx, j_idx in SKELETON:  # Ya son 0-based
        if vis[i_idx] <= 0 or vis[j_idx] <= 0:
            continue
        
        src = keypoints[i_idx]
        dst = keypoints[j_idx]
        
        # Calcular ki, kj para determinar colores
        ki = min(i_idx, j_idx)
        kj = max(i_idx, j_idx)
        
        # head-to-head thin pink line (EXACTO de Visualizer)
        if ki < 5 and kj < 5:
            color = (250, 32, 98)  # [250/255, 32/255, 98/255] * 255
            line_width = max(1, int(1 * scale * line_width_multiplier))
        # body thick colored segments (EXACTO de Visualizer)
        elif ki >= 5 and kj >= 5:
            pair = (ki, kj)
            if pair in GREEN:
                color = (38, 252, 145)  # (38/255, 252/255, 145/255) * 255
            elif pair in YELLOW:
                color = (250, 244, 60)  # (250/255, 244/255, 60/255) * 255
            elif pair in BLUE:
                color = (104, 252, 252)  # (104/255, 252/255, 252/255) * 255
            elif pair in PINK:
                color = (255, 148, 212)  # (255/255, 148/255, 212/255) * 255
            else:
                continue  # No dibujar si no está en los pares definidos
            line_width = max(2, int(6.0 * scale * line_width_multiplier))
        else:
            # Casos mixtos - no dibujar según Visualizer
            continue
        
        draw.line([tuple(src), tuple(dst)], fill=color, width=line_width)
    
    # Dibujar círculos para keypoints EXACTAMENTE como Visualizer
    # Separar head y body como en Visualizer
    vis_head = keypoints[:5][vis[:5] > 0]
    vis_body = keypoints[5:][vis[5:] > 0]
    
    # Dibujar círculos de cabeza (más pequeños)
    for pnt in vis_head:
        radius = int(1.5 * scale * 1.2 * point_size_multiplier)
        draw.ellipse([pnt[0]-radius, pnt[1]-radius, pnt[0]+radius, pnt[1]+radius], 
                    outline=(0, 0, 0), width=2)
    
    # Dibujar círculos de cuerpo (más grandes)
    for pnt in vis_body:
        radius = int(3.0 * scale * 1.2 * point_size_multiplier)
        draw.ellipse([pnt[0]-radius, pnt[1]-radius, pnt[0]+radius, pnt[1]+radius], 
                    outline=(0, 0, 0), width=2)
    
    return img_pil


def get_pose_keypoints(person_crop_pil, bbox_expanded, pose_model, device, conf):
    """
    Estima los keypoints de una persona a partir de su recorte y los devuelve en
    coordenadas de la imagen original.

    PROCESO DE TRANSFORMACIÓN DE COORDENADAS:
    1. Imagen recortada → Transformación afín → Espacio del modelo (512x384)
    2. Predicción del modelo → Coordenadas en espacio del modelo
    3. Transformación inversa → Espacio del recorte original  
    4. Suma offset → Espacio de la imagen completa
    
    Esto asegura que los keypoints se dibujen en la posición correcta.
    Sin la transformación inversa correcta, los keypoints no se alinean con las personas.

    Args:
        person_crop_pil (PIL.Image): Imagen PIL RGB del recorte de la persona
        bbox_expanded (list): Caja [x_min, y_min, x_max, y_max] del recorte en coordenadas de la imagen
        pose_model (torch.nn.Module): Modelo de pose (PyTorch) en modo evaluación
        device (torch.device): 'cuda' o 'cpu' según disponibilidad
        conf: Configuración del modelo (usada por el post-procesado)

    Returns:
        tuple: (keypoints, scores, plot_scale) donde:
            keypoints (np.array): Array de shape (17, 2) con coordenadas absolutas en la imagen original
            scores (np.array): Array de shape (17, 1) con confianza asociada a cada keypoint
            plot_scale (float): Factor recomendado para escalar elementos de dibujo
    """
    # Obtener tamaño del modelo
    model_w, model_h = MODEL_W, MODEL_H
    # Obtener dimensiones del recorte
    crop_w, crop_h = person_crop_pil.size
    x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded = bbox_expanded
    
    # Crear metadatos de centro y escala a partir del tamaño del recorte
    center, scale = make_meta_from_wh(crop_w, crop_h, use_max_scale=True)
    
    # Preprocesar con transformación afín al tamaño esperado por el modelo
    input_patch = preprocess_patch(person_crop_pil, center, scale, (model_w, model_h))
    
    # Forward pass original
    input_tensor = normalize(Image.fromarray(input_patch)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = pose_model(input_tensor)
    
    # Forward pass flipped
    input_flipped = np.flip(input_patch, axis=1)
    input_flipped_tensor = normalize(Image.fromarray(input_flipped)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs_flipped = pose_model(input_flipped_tensor)
    
    # Flip pairs para COCO
    flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                  [9, 10], [11, 12], [13, 14], [15, 16]]
    
    # Obtener predicciones
    c_arr = center[None, :].astype(np.float32)
    s_arr = scale[None, :].astype(np.float32)
    
    preds, maxvals, preds_raw = get_final_preds_match(conf, outputs, c_arr, s_arr)
    preds_f, maxvals_f, preds_raw_f = get_final_preds_match(conf, outputs_flipped, c_arr, s_arr, flip_pairs)
    
    # Promediar predicciones
    preds_raw = (preds_raw + preds_raw_f) / 2.0
    maxvals = (maxvals + maxvals_f) / 2.0
    
    preds_final = np.asarray(preds_raw[0])  # (17, 2) - coordenadas del recorte
    heat_conf = np.asarray(maxvals[0, :, 0])  # (17,)
    
    # Aplicar filtrado
    conf_th = 0.8
    valid_mask = np.zeros((17,), dtype=np.int32)

    logits = outputs['pred_logits']
    pred_coords_q = outputs['pred_coords']
    logits_f = outputs_flipped['pred_logits']
    pred_coords_q_f = outputs_flipped['pred_coords']

    probs = torch.softmax((logits + logits_f) / 2.0, dim=-1)[0].cpu().numpy()
    coords_q = ((pred_coords_q + pred_coords_q_f) / 2.0)[0].cpu().numpy()
    coords_q *= np.array([model_w, model_h])
    

    ACCEPT_PROB_TH = 0.3
    ACCEPT_DIST_TH = 50.0

    best_probs = []
    for j in range(17):

        class_probs = probs[:, j]
        bg_probs = probs[:, 17]  # fondo

        # Costo de matching basado en probabilidad (entre más alta la probabilidad, menor el costo)
        cost = -class_probs

        # Ignorar queries donde la probabilidad del fondo sea mayor que la del keypoint
        valid_queries = np.where((class_probs > bg_probs) & (bg_probs > 0.8))[0]
        if len(valid_queries) == 0:
            # Si todos los queries creen que es fondo, elegimos el de menor costo global
            best_query_idx = np.argmin(cost)
            best_prob = class_probs[best_query_idx]
            best_probs.append(best_prob)
        else:
            # Entre los válidos, seleccionamos el de menor costo (mayor probabilidad)
            best_query_idx = valid_queries[np.argmin(cost[valid_queries])]
            best_prob = class_probs[best_query_idx]

        # Coordenadas
        query_coord = coords_q[best_query_idx]
        joint_coord = preds_final[j]
        distance = np.linalg.norm(query_coord - joint_coord)

        # Criterios de aceptación
        accepted = (
            best_prob > ACCEPT_PROB_TH and
            distance < ACCEPT_DIST_TH and
            heat_conf[j] > conf_th
        )
        valid_mask[j] = 1 if accepted else 0
    
    # Filtrar keypoints (necesitamos copia porque preds_final se usa en el loop anterior)
    filtered_preds = preds_final.copy()
    filtered_scores = heat_conf.copy()
    
    for i in range(17):
        if valid_mask[i] == 0:
            filtered_preds[i] = [np.nan, np.nan]
            filtered_scores[i] = 0.0

    trans_inv = get_affine_transform(center, scale, 0, (model_w, model_h), inv=1)
    
    keypoints_crop_space = []
    for kp in filtered_preds:
        if not np.isnan(kp[0]):
            kp_homogeneous = np.array([kp[0], kp[1], 1.0])
            kp_original = trans_inv.dot(kp_homogeneous)
            keypoints_crop_space.append([kp_original[0], kp_original[1]])
        else:
            keypoints_crop_space.append([np.nan, np.nan])
    
    keypoints_crop_space = np.array(keypoints_crop_space)
    
    # Transformar del espacio del recorte al espacio de la imagen original
    keypoints_original = keypoints_crop_space.copy()
    keypoints_original[:, 0] += x_min_expanded
    keypoints_original[:, 1] += y_min_expanded
    
    # Calcular plot_scale
    plot_scale = np.linalg.norm(scale) / 2.0
    
    return keypoints_original, filtered_scores.reshape(-1, 1), plot_scale


def infer_multi_person_pose(image_pil, pose_model, detr_model, detr_processor, device, conf, 
                           point_size_multiplier=1.0, line_width_multiplier=1.0,
                           detr_threshold=0.9):
    """
    Inferencia multi-persona usando DETR para detectar personas y luego keypoints.

    FLUJO MULTI-PERSONA:
    1. DETR detecta todas las personas en la imagen (usa matching bipartito, no requiere NMS)
    2. Para cada persona detectada:
       - Expandimos la bbox para dar contexto extra para keypoints en bordes
       - Recortamos y preprocesamos la región
       - Ejecutamos el modelo de pose
       - Transformamos coordenadas al espacio original
       - Dibujamos keypoints y esqueleto
    
    Args:
        image_pil (PIL.Image): Imagen RGB completa
        pose_model (torch.nn.Module): Modelo de pose PRTR-HRNet
        detr_model: Modelo DETR para detección de personas
        detr_processor: Procesador DETR para pre/post-procesado
        device (torch.device): Dispositivo de cómputo
        conf: Configuración del modelo
        point_size_multiplier (float): Multiplicador para tamaño de puntos visualizados
        line_width_multiplier (float): Multiplicador para grosor de líneas del esqueleto
        detr_threshold (float): Umbral de confianza para detección DETR (0.9 recomendado)

    Returns:
        tuple: (image_with_keypoints, person_count, persons_details) donde:
            image_with_keypoints (PIL.Image): Imagen con keypoints dibujados
            person_count (int): Número de personas detectadas
            persons_details (list[dict]): Lista con detalles por persona: bbox, keypoints (17x2) y confidences (17,)
    """
    try:
        # Detect persones con DETR
        inputs = detr_processor(images=image_pil, return_tensors="pt")
        outputs = detr_model(**inputs)
        
        target_sizes = torch.tensor([image_pil.size[::-1]])
        # DETR usa matching bipartito, no requiere NMS. El threshold filtra por confianza
        results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=detr_threshold)[0]
        
        # Filtrar solo detecciones de personas (label == 1 en COCO dataset)
        # Ordenar por score descendente para priorizar detecciones más confiables
        persons = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # En COCO dataset, label 1 es "person"
            if label.item() == 1:  # Solo personas
                persons.append((score, box))
        
        # Ordenar por score descendente (DETR ya maneja duplicados con matching bipartito)
        persons.sort(key=lambda x: x[0].item(), reverse=True)
        
        # Crear una copia de la imagen original donde dibujaremos todos los keypoints
        image_with_keypoints = image_pil.copy()
        
        # Procesar cada persona detectada
        person_count = 0
        persons_details = []
        for idx, (score, box) in enumerate(persons, 1):
            person_count += 1
            
            # Convertir coordenadas: [x_min, y_min, x_max, y_max]
            box = box.tolist()
            
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            # Expandir la caja 50% (25% en cada lado)
            # La expansión de bbox ayuda con keypoints en bordes: dar contexto extra alrededor
            # de la persona mejora la detección de keypoints cerca de los bordes del recorte.
            EXPANSION_FACTOR = 0.125  # 12.5% en cada dirección = 25% total
            expand_x = width * EXPANSION_FACTOR
            expand_y = height * EXPANSION_FACTOR
            
            # Nuevas coordenadas expandidas
            x_min_expanded = max(0, x_min - expand_x)  # Asegurar que no sea negativo
            y_min_expanded = max(0, y_min - expand_y)
            x_max_expanded = min(image_pil.size[0], x_max + expand_x)  # Asegurar que no exceda la imagen
            y_max_expanded = min(image_pil.size[1], y_max + expand_y)
            
            # Recortar la persona de la imagen usando la caja expandida
            person_crop = image_pil.crop((int(x_min_expanded), int(y_min_expanded), int(x_max_expanded), int(y_max_expanded)))
            
            # Obtener keypoints de pose para esta persona (en coordenadas del recorte)
            try:
                keypoints_absolute, keypoint_scores_crop, plot_scale = get_pose_keypoints(
                    person_crop,
                    [x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded],
                    pose_model, device, conf
                )

                # Aplanar scores una sola vez para reutilizar
                keypoint_scores_flat = keypoint_scores_crop.flatten()

                # Dibujar keypoints sobre la imagen original
                image_with_keypoints = draw_keypoints_pil(image_with_keypoints, keypoints_absolute, 
                                                        confidences=keypoint_scores_flat, 
                                                        scale=plot_scale,
                                                        point_size_multiplier=point_size_multiplier,
                                                        line_width_multiplier=line_width_multiplier)

                # Guardar detalles por persona
                persons_details.append({
                    'bbox': [x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded],
                    'keypoints': keypoints_absolute.tolist(),
                    'confidences': keypoint_scores_flat.tolist()
                })
                
            except Exception as e:
                st.warning(f"Error procesando keypoints para persona {person_count}: {e}")
        
        return image_with_keypoints, person_count, persons_details
        
    except Exception as e:
        st.error(f"Error en inferencia multi-persona: {e}")
        return None, 0, []

def process_single_frame(frame_data, pose_model, detr_model, detr_processor, device, conf,
                        point_size_multiplier, line_width_multiplier, detr_threshold,
                        width, height):
    """
    Procesa un solo frame. Función auxiliar para procesamiento en paralelo.
    
    Args:
        frame_data: tuple (frame_idx, frame_pil, frame_bgr)
        pose_model: Modelo de pose
        detr_model: Modelo DETR
        detr_processor: Procesador DETR
        device: Dispositivo de cómputo
        conf: Configuración del modelo
        point_size_multiplier: Multiplicador para tamaño de puntos
        line_width_multiplier: Multiplicador para grosor de líneas
        detr_threshold: Umbral de confianza DETR
        width: Ancho del frame
        height: Alto del frame
    
    Returns:
        tuple: (frame_idx, result_bgr) o (frame_idx, None) si hay error
    """
    frame_idx, frame_pil, frame_bgr = frame_data
    try:
        # Aplicar inferencia de pose
        result_image, person_count, _ = infer_multi_person_pose(
            frame_pil, pose_model, detr_model, detr_processor, device, conf,
            point_size_multiplier, line_width_multiplier, detr_threshold
        )
        
        if result_image is not None:
            # Convertir PIL a OpenCV (ya viene del tamaño correcto porque el frame fue redimensionado antes)
            result_np = np.array(result_image)
            # Solo redimensionar si es absolutamente necesario (no debería ser necesario)
            if result_np.shape[:2] != (height, width):
                result_np = cv2.resize(result_np, (width, height), interpolation=cv2.INTER_LINEAR)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            return (frame_idx, result_bgr)
        else:
            # Si falla, devolver frame original
            return (frame_idx, frame_bgr)
            
    except Exception as e:
        # Si hay error, devolver frame original
        return (frame_idx, frame_bgr)

def get_video_hash(video_bytes):
    """Genera un hash del video para usar como clave de caché"""
    return hashlib.md5(video_bytes).hexdigest()

def resize_frame_if_needed(frame, target_width, target_height):
    """
    Redimensiona un frame al tamaño objetivo usando interpolación rápida.
    Para velocidad, simplemente redimensiona sin mantener aspect ratio exacto.
    
    Args:
        frame: Frame OpenCV (BGR)
        target_width: Ancho objetivo
        target_height: Alto objetivo
    
    Returns:
        Frame redimensionado
    """
    h, w = frame.shape[:2]
    if w == target_width and h == target_height:
        return frame
    
    # Redimensionar directamente al tamaño objetivo (más rápido)
    # Usar INTER_LINEAR para balance velocidad/calidad
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized


def process_video(video_file, pose_model, detr_model, detr_processor, device, conf,
                 point_size_multiplier=1.0, line_width_multiplier=1.0,
                 detr_threshold=0.9, max_frames=None, frame_skip=1,
                 batch_size=8, target_resolution=None):
    """
    Procesa un video frame por frame aplicando detección de pose en streaming.
    Procesa y escribe frames inmediatamente para reducir uso de memoria.
    
    Args:
        video_file: Archivo de video subido (BytesIO o path)
        pose_model: Modelo de pose
        detr_model: Modelo DETR
        detr_processor: Procesador DETR
        device: Dispositivo de cómputo
        conf: Configuración del modelo
        point_size_multiplier: Multiplicador para tamaño de puntos
        line_width_multiplier: Multiplicador para grosor de líneas
        detr_threshold: Umbral de confianza DETR
        max_frames: Número máximo de frames a procesar (None = todos)
        frame_skip: Procesar cada N frames (1 = todos, 2 = cada 2, etc.)
        batch_size: Número de frames a procesar en paralelo (default: 8, reducido para menor uso de memoria)
        target_resolution: tuple (width, height) para redimensionar frames, o None para mantener original
    
    Returns:
        tuple: (video_path, total_frames, processed_frames, fps, width, height)
    """
    tfile = None
    output_path = None
    cap = None
    out = None
    
    try:
        # Leer video completo para hash y guardar
        video_bytes = video_file.read()
        video_hash = get_video_hash(video_bytes)
        
        # Incluir resolución en el hash del caché
        res_str = f"{target_resolution[0]}x{target_resolution[1]}" if target_resolution else "original"
        cache_hash = hashlib.md5(f"{video_hash}_{res_str}_{batch_size}_{frame_skip}".encode()).hexdigest()
        
        # Verificar caché
        cache_dir = tempfile.gettempdir()
        cache_file = os.path.join(cache_dir, f"pose_video_cache_{cache_hash}.mp4")
        
        if os.path.exists(cache_file):
            st.info(f"Video encontrado en caché. Usando versión procesada anteriormente.")
            # Leer propiedades del video en caché
            cap_cache = cv2.VideoCapture(cache_file)
            if cap_cache.isOpened():
                fps = int(cap_cache.get(cv2.CAP_PROP_FPS)) or 30
                width = int(cap_cache.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap_cache.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap_cache.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_cache.release()
                return cache_file, total_frames, total_frames, fps, width, height
        
        # Guardar video temporalmente
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_bytes)
        tfile.close()
        
        # Abrir video
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            if tfile and os.path.exists(tfile.name):
                os.unlink(tfile.name)
            return None, 0, 0, 0, 0, 0
        
        # Obtener propiedades del video original
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if orig_width == 0 or orig_height == 0:
            if tfile and os.path.exists(tfile.name):
                os.unlink(tfile.name)
            return None, 0, 0, 0, 0, 0
        
        # Aplicar resolución objetivo si se especifica
        if target_resolution:
            width, height = target_resolution
            st.info(f"Redimensionando video de {orig_width}x{orig_height} a {width}x{height} para procesamiento más rápido")
        else:
            width, height = orig_width, orig_height
        
        # Crear video de salida (usar caché si existe)
        output_path = cache_file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            if tfile and os.path.exists(tfile.name):
                os.unlink(tfile.name)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
            return None, 0, 0, 0, 0, 0
        
        # Procesar en streaming: leer, procesar en batches pequeños, escribir en orden
        frame_count = 0
        processed_count = 0
        frames_to_write = {}  # Buffer ordenado para escribir frames en orden: {frame_idx: frame_bgr}
        next_write_idx = 0  # Índice del siguiente frame a escribir
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Cola para frames a procesar (limitar tamaño para reducir memoria)
        frame_queue = deque(maxlen=batch_size * 2)
        
        # Procesar frames en streaming
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Redimensionar frame si es necesario (ANTES de procesar para mayor velocidad)
            if target_resolution:
                frame = resize_frame_if_needed(frame, width, height)
            
            # Limitar número de frames si se especifica
            if max_frames is not None and processed_count >= max_frames:
                # Guardar frame sin procesar en buffer para escribir en orden
                frames_to_write[frame_count] = frame
                frame_count += 1
                continue
            
            # Saltar frames según frame_skip
            if frame_count % frame_skip != 0:
                # Guardar frame sin procesar en buffer para escribir en orden
                frames_to_write[frame_count] = frame
                frame_count += 1
                continue
            
            # Convertir frame a PIL Image (ya redimensionado)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Agregar a cola para procesamiento
            frame_queue.append((frame_count, frame_pil, frame))
            
            frame_count += 1
            
            # Procesar batch cuando la cola esté llena
            if len(frame_queue) >= batch_size:
                # Procesar batch actual
                batch_frames = list(frame_queue)
                frame_queue.clear()
                
                # Procesar frames en paralelo
                batch_results = process_batch_frames(
                    batch_frames, pose_model, detr_model, detr_processor, device, conf,
                    point_size_multiplier, line_width_multiplier, detr_threshold,
                    width, height, max_workers=min(batch_size, len(batch_frames))
                )
                
                # Guardar todos los resultados del batch en el buffer
                for frame_idx, result_bgr in batch_results.items():
                    frames_to_write[frame_idx] = result_bgr
                
                # Escribir frames procesados en orden (escribir todos los que estén disponibles consecutivamente)
                while next_write_idx in frames_to_write:
                    out.write(frames_to_write.pop(next_write_idx))
                    next_write_idx += 1
                    processed_count += 1
                
                # Actualizar progreso
                if total_frames > 0:
                    progress = min(1.0, frame_count / total_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"Procesando: Frame {frame_count}/{total_frames} ({processed_count} procesados, {len(frames_to_write)} en buffer)")
                else:
                    progress_bar.progress(0.5)
                    status_text.text(f"Procesando: Frame {frame_count} ({processed_count} procesados, {len(frames_to_write)} en buffer)")
            
            # Limitar frames procesados
            if max_frames is not None and processed_count >= max_frames:
                # Leer frames restantes y guardarlos en buffer
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if target_resolution:
                        frame = resize_frame_if_needed(frame, width, height)
                    frames_to_write[frame_count] = frame
                    frame_count += 1
                break
        
        # Procesar frames restantes en la cola
        if len(frame_queue) > 0:
            batch_frames = list(frame_queue)
            frame_queue.clear()
            
            # Procesar frames restantes
            batch_results = process_batch_frames(
                batch_frames, pose_model, detr_model, detr_processor, device, conf,
                point_size_multiplier, line_width_multiplier, detr_threshold,
                width, height, max_workers=min(batch_size, len(batch_frames))
            )
            
            for frame_idx, result_bgr in batch_results.items():
                frames_to_write[frame_idx] = result_bgr
        
        # Escribir TODOS los frames restantes en orden
        # Primero escribir todos los frames consecutivos disponibles
        while next_write_idx in frames_to_write:
            out.write(frames_to_write.pop(next_write_idx))
            next_write_idx += 1
            processed_count += 1
        
        # Si aún quedan frames en el buffer (puede haber gaps), escribir los restantes en orden
        if frames_to_write:
            # Ordenar los índices restantes y escribir en orden
            remaining_indices = sorted(frames_to_write.keys())
            for frame_idx in remaining_indices:
                out.write(frames_to_write.pop(frame_idx))
                processed_count += 1
        
        progress_bar.progress(1.0)
        status_text.text(f"Procesamiento completado: {processed_count} frames procesados")
        
    except Exception as e:
        st.error(f"Error procesando video: {e}")
        import traceback
        st.code(traceback.format_exc())
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)
        return None, 0, 0, 0, 0, 0
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if tfile and os.path.exists(tfile.name):
            os.unlink(tfile.name)
    
    return output_path, total_frames, processed_count, fps, width, height


# ---------- Funciones auxiliares para UI ----------
def show_image_metrics(image):
    """Muestra las métricas de tamaño de una imagen"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{image.size[0]} x {image.size[1]}</div>
        <div class="metric-label">píxeles</div>
    </div>
    """, unsafe_allow_html=True)

def show_result_with_download(result_image, person_count, file_name_prefix, file_extension="png"):
    """Muestra el resultado de la inferencia con opción de descarga"""
    st.image(result_image, caption="Pose Detection Result", width='stretch')
    
    # Estadísticas
    st.markdown(f"""
    <div class=\"metric-card\">
        <div class=\"metric-value\">{person_count}</div>
        <div class=\"metric-label\">personas detectadas</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Opción de descarga
    buf = io.BytesIO()
    result_image.save(buf, format=file_extension.upper())
    byte_im = buf.getvalue()
    
    st.download_button(
        label="Descargar Resultado",
        data=byte_im,
        file_name=f"pose_result_{file_name_prefix}.{file_extension}",
        mime=f"image/{file_extension}"
    )

def process_and_show_pose(image, pose_model, detr_model, detr_processor, device, conf,
                          point_size_multiplier, line_width_multiplier, detr_threshold,
                          file_name_prefix, processing_message="Procesando..."):
    """Procesa una imagen y muestra el resultado"""
    with st.spinner(processing_message):
        result_image, person_count, persons_details = infer_multi_person_pose(
            image, pose_model, detr_model, detr_processor, device, conf,
            point_size_multiplier, line_width_multiplier, detr_threshold
        )
        
        if result_image is not None:
            show_result_with_download(result_image, person_count, file_name_prefix)
        else:
            st.error("No se pudo procesar la imagen")
        return result_image, person_count, persons_details

def process_batch_frames(batch_frames, pose_model, detr_model, detr_processor, device, conf,
                        point_size_multiplier, line_width_multiplier, detr_threshold,
                        width, height, max_workers=None):
    """Procesa un batch de frames en paralelo"""
    if max_workers is None:
        max_workers = min(len(batch_frames), 8)  # Limitar workers por defecto
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame = {
            executor.submit(
                process_single_frame,
                (idx, pil, frame_bgr),
                pose_model, detr_model, detr_processor, device, conf,
                point_size_multiplier, line_width_multiplier,
                detr_threshold, width, height
            ): idx for idx, pil, frame_bgr in batch_frames
        }
        
        batch_results = {}
        for future in as_completed(future_to_frame):
            try:
                frame_idx, result_bgr = future.result()
                batch_results[frame_idx] = result_bgr
            except Exception as e:
                frame_idx = future_to_frame[future]
                # Usar frame original si hay error
                original_frame = next((frame_bgr for idx, _, frame_bgr in batch_frames if idx == frame_idx), None)
                if original_frame is not None:
                    batch_results[frame_idx] = original_frame
        
        return batch_results


def main():
    # Header principal con diseño mejorado
    st.markdown("""
    <div class="main-header">
        <h1>Multi-Person Pose Detection</h1>
        <p>Detección inteligente de poses humanas con IA avanzada</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar con información y configuración
    with st.sidebar:
        st.markdown("### Diseño Visual")
        st.markdown("""
        <div class="card">
        <strong>Visualización elegante del esqueleto humano</strong><br><br>
        
        <strong>Verde:</strong> Brazo izquierdo<br>
        <strong>Amarillo:</strong> Brazo derecho<br>
        <strong>Azul:</strong> Pierna izquierda<br>
        <strong>Rosa:</strong> Pierna derecha<br>
        <strong>Rosa claro:</strong> Conexiones de cabeza<br><br>
        
        <em>Diseño colorido y profesional para una visualización clara de las poses detectadas.</em>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Ajustes")
        
        # Contenedor para el slider de líneas
        st.markdown('<div class="slider-container">', unsafe_allow_html=True)
        line_width_multiplier = st.slider(
            "Grosor de líneas del esqueleto",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Ajusta el grosor de las líneas del esqueleto"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        point_size_multiplier = st.slider(
            "Tamaño de puntos",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Ajusta el tamaño de los puntos"
        )
        
        detr_threshold = st.slider(
            "Sensibilidad de detección",
            min_value=0.1,
            max_value=0.99,
            value=0.9,
            step=0.05,
            help="Controla qué tan estricta es la detección de personas"
        )
        
    
    # Cargar modelos
    with st.spinner("Cargando modelos..."):
        pose_model, detr_model, detr_processor, device, conf = load_models()
    
    if pose_model is None or detr_model is None:
        st.error("No se pudieron cargar los modelos. Verifica que los archivos estén disponibles.")
        return
    
    # Crear pestañas usando radio buttons
    st.markdown("### Modo de Análisis")
    mode = st.radio(
        "Selecciona el modo de análisis:",
        ["Análisis de Imagen", "Tomar Foto", "Análisis de Video"]
    )
    
    if mode == "Análisis de Imagen":
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Subir Imagen")
            uploaded_file = st.file_uploader(
                "Selecciona una imagen",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Formatos soportados: JPG, JPEG, PNG, BMP"
            )
            
            if uploaded_file is not None:
                # Mostrar imagen original
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Imagen Original", width='stretch')
                show_image_metrics(image)
        
        with col2:
            st.markdown("#### Resultado")
            
            if uploaded_file is not None:
                if st.button("Analizar Pose", key="analyze_image"):
                    process_and_show_pose(
                        image, pose_model, detr_model, detr_processor, device, conf,
                        point_size_multiplier, line_width_multiplier, detr_threshold,
                        uploaded_file.name, "Procesando imagen..."
                    )
            else:
                st.markdown("""
                <div class=\"camera-container\">
                    <h4>Sube una imagen para comenzar</h4>
                </div>
                """, unsafe_allow_html=True)
    
    elif mode == "Tomar Foto":
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Capturar Foto")
            
            # Widget de cámara de Streamlit
            camera_photo = st.camera_input("Toma una foto para analizar la pose")
            
            if camera_photo is not None:
                # Convertir la foto a PIL Image
                image = Image.open(camera_photo).convert("RGB")
                
                # Mostrar imagen capturada
                st.image(image, caption="Foto Capturada", width='stretch')
                show_image_metrics(image)
                
                # Botón de análisis en la misma columna
                if st.button("Analizar Pose", key="analyze_camera_photo"):
                    file_name = f"camera_{int(time.time())}"
                    process_and_show_pose(
                        image, pose_model, detr_model, detr_processor, device, conf,
                        point_size_multiplier, line_width_multiplier, detr_threshold,
                        file_name, "Procesando foto..."
                    )
            else:
                st.markdown("""
                <div class=\"camera-container\">
                    <h4>Toma una foto para comenzar</h4>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Instrucciones")
            st.markdown("""
            <div class=\"camera-container\">
                <h4>Cómo usar la cámara</h4>
                <ol>
                    <li>Haz clic en "Toma una foto"</li>
                    <li>Permite el acceso a la cámara</li>
                    <li>Posiciona la persona en el marco</li>
                    <li>Presiona "Analizar Pose"</li>
                </ol>
                <p><strong>Consejos:</strong></p>
                <ul>
                    <li>Buena iluminación</li>
                    <li>Persona completa visible</li>
                    <li>Fondo simple</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    elif mode == "Análisis de Video":
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Subir Video")
            uploaded_video = st.file_uploader(
                "Selecciona un video",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                help="Formatos soportados: MP4, AVI, MOV, MKV, WEBM"
            )
            
            if uploaded_video is not None:
                # Mostrar información del video
                st.video(uploaded_video)
                
                # Configuración de procesamiento
                st.markdown("#### Configuración de Procesamiento")
                
                max_frames_option = st.selectbox(
                    "Límite de frames",
                    ["Todos los frames", "50 frames", "100 frames", "200 frames", "500 frames"],
                    help="Limita el número de frames a procesar para videos largos"
                )
                
                max_frames_map = {
                    "Todos los frames": None,
                    "50 frames": 50,
                    "100 frames": 100,
                    "200 frames": 200,
                    "500 frames": 500
                }
                max_frames = max_frames_map[max_frames_option]
                
                frame_skip = st.slider(
                    "Procesar cada N frames",
                    min_value=1,
                    max_value=10,
                    value=1,
                    help="1 = todos los frames, 2 = cada 2 frames, etc. Útil para videos largos"
                )
                
                batch_size = st.slider(
                    "Tamaño del batch (procesamiento paralelo)",
                    min_value=1,
                    max_value=16,
                    value=8,
                    step=1,
                    help="Número de frames a procesar en paralelo. Valores más altos = más rápido pero más uso de memoria. Recomendado: 4-8 para reducir uso de memoria. El video se procesa en streaming para optimizar recursos."
                )
                
                # Opción de resolución para procesamiento más rápido
                resolution_option = st.selectbox(
                    "Resolución de procesamiento (para mayor velocidad)",
                    ["Original", "1080p (1920x1080)", "720p (1280x720)", "480p (854x480)", "360p (640x360)", "240p (426x240)"],
                    index=2,  # Por defecto 720p para balance velocidad/calidad
                    help="Reducir la resolución acelera significativamente el procesamiento. Recomendado: 720p o 480p para mejor velocidad."
                )
                
                # Obtener resolución objetivo
                resolution_map = {
                    "Original": None,
                    "1080p (1920x1080)": (1920, 1080),
                    "720p (1280x720)": (1280, 720),
                    "480p (854x480)": (854, 480),
                    "360p (640x360)": (640, 360),
                    "240p (426x240)": (426, 240)
                }
                target_resolution = resolution_map[resolution_option]
                
                st.info(f"""
                **Configuración:** {max_frames_option} | Frame skip: {frame_skip} | Resolución: {resolution_option}
                
                **Optimizado para velocidad y eficiencia**
                """)
        
        with col2:
            st.markdown("#### Resultado")
            
            if uploaded_video is not None:
                if st.button("Procesar Video", key="process_video"):
                    # Resetear el stream del archivo
                    uploaded_video.seek(0)
                    
                    with st.spinner("Procesando video... Esto puede tardar varios minutos."):
                        try:
                            # Crear un BytesIO wrapper para el archivo
                            video_bytes_io = io.BytesIO(uploaded_video.read())
                            uploaded_video.seek(0)  # Resetear para mostrar el video original
                            
                            output_path, total_frames, processed_frames, fps, width, height = process_video(
                                video_bytes_io,
                                pose_model, detr_model, detr_processor, device, conf,
                                point_size_multiplier, line_width_multiplier,
                                detr_threshold,
                                max_frames=max_frames, frame_skip=frame_skip,
                                batch_size=batch_size, target_resolution=target_resolution
                            )
                            
                            if output_path and os.path.exists(output_path):
                                # Leer el video procesado
                                with open(output_path, 'rb') as video_file:
                                    video_bytes = video_file.read()
                                
                                # Mostrar estadísticas
                                st.markdown(f"""
                                <div class=\"metric-card\">
                                    <div class=\"metric-value\">{processed_frames}</div>
                                    <div class=\"metric-label\">frames procesados</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                
                                # Mostrar video procesado
                                st.video(video_bytes)
                                
                                # Opción de descarga
                                st.download_button(
                                    label="Descargar Video Procesado",
                                    data=video_bytes,
                                    file_name=f"pose_result_{uploaded_video.name}",
                                    mime="video/mp4"
                                )
                                
                                # Limpiar archivo temporal después de un tiempo
                                # (Streamlit manejará la limpieza cuando se recargue la página)
                            else:
                                st.error("No se pudo procesar el video. Verifica que el formato sea compatible.")
                                
                        except Exception as e:
                            st.error(f"Error procesando el video: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.markdown("""
                <div class=\"camera-container\">
                    <h4>Sube un video para comenzar</h4>
                    <p>Selecciona un video desde tu dispositivo para analizar las poses</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Multi-Person Pose Detection </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
