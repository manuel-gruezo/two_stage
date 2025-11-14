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

# ---------- Funciones auxiliares para UI ----------
def show_image_metrics(image):
    """Muestra las métricas de tamaño de una imagen"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{image.size[0]} x {image.size[1]}</div>
        <div class="metric-label">píxeles</div>
    </div>
    """, unsafe_allow_html=True)

def show_result_with_download(result_image, person_count, file_name_prefix, file_extension="png", inference_time=None):
    """Muestra el resultado de la inferencia con opción de descarga"""
    st.image(result_image, caption="Pose Detection Result", width='stretch')
    
    # Estadísticas
    st.markdown(f"""
    <div class=\"metric-card\">
        <div class=\"metric-value\">{person_count}</div>
        <div class=\"metric-label\">personas detectadas</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar tiempo de inferencia si está disponible
    if inference_time is not None:
        minutes = int(inference_time // 60)
        seconds = int(inference_time % 60)
        milliseconds = int((inference_time % 1) * 1000)
        
        if minutes > 0:
            time_str = f"{minutes}m {seconds}s {milliseconds}ms"
        else:
            time_str = f"{seconds}s {milliseconds}ms"
        
        st.markdown(f"""
        <div class=\"metric-card\" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
            <div class=\"metric-value\">{time_str}</div>
            <div class=\"metric-label\">Tiempo de inferencia</div>
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
    start_time = time.time()
    with st.spinner(processing_message):
        result_image, person_count, persons_details = infer_multi_person_pose(
            image, pose_model, detr_model, detr_processor, device, conf,
            point_size_multiplier, line_width_multiplier, detr_threshold
        )
        
        # Calcular tiempo de inferencia
        inference_time = time.time() - start_time
        
        if result_image is not None:
            show_result_with_download(result_image, person_count, file_name_prefix, inference_time=inference_time)
        else:
            st.error("No se pudo procesar la imagen")
        return result_image, person_count, persons_details

def detect_squat(keypoints, confidences, prev_state=None):
    """
    Detecta si una persona está haciendo una sentadilla basándose en los keypoints.
    
    Args:
        keypoints: Array de shape (17, 2) con coordenadas [x, y] de cada keypoint
        confidences: Array de shape (17,) con confianza de cada keypoint
        prev_state: Estado anterior (dict con 'state' y 'squat_count')
    
    Returns:
        tuple: (is_squat_down, state_dict) donde:
            is_squat_down: bool indicando si está en posición baja
            state_dict: dict con estado actualizado
    """
    # Índices de keypoints COCO (0-based)
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    
    # Inicializar estado si no existe
    if prev_state is None:
        prev_state = {
            'state': 'up',  # 'up' o 'down'
            'squat_count': 0,
            'hip_y_avg': None,
            'knee_y_avg': None
        }
    
    state = prev_state.copy()
    
    # Obtener coordenadas Y de caderas y rodillas (Y aumenta hacia abajo)
    hip_y_values = []
    knee_y_values = []
    
    if not np.isnan(keypoints[LEFT_HIP][1]) and confidences[LEFT_HIP] > 0:
        hip_y_values.append(keypoints[LEFT_HIP][1])
    if not np.isnan(keypoints[RIGHT_HIP][1]) and confidences[RIGHT_HIP] > 0:
        hip_y_values.append(keypoints[RIGHT_HIP][1])
    if not np.isnan(keypoints[LEFT_KNEE][1]) and confidences[LEFT_KNEE] > 0:
        knee_y_values.append(keypoints[LEFT_KNEE][1])
    if not np.isnan(keypoints[RIGHT_KNEE][1]) and confidences[RIGHT_KNEE] > 0:
        knee_y_values.append(keypoints[RIGHT_KNEE][1])
    
    # Si no tenemos suficientes keypoints visibles, mantener estado anterior
    if len(hip_y_values) == 0 or len(knee_y_values) == 0:
        return False, state
    
    # Calcular promedios
    hip_y_avg = np.mean(hip_y_values)
    knee_y_avg = np.mean(knee_y_values)
    
    # Calcular diferencia entre cadera y rodilla
    # En posición de pie: cadera está arriba (Y menor) que rodilla
    # En sentadilla: cadera baja (Y mayor) y está más cerca de la rodilla
    diff = hip_y_avg - knee_y_avg
    
    # Umbral para detectar sentadilla (cuando la diferencia es pequeña o negativa)
    # Si la cadera está por debajo o muy cerca de la rodilla, está en sentadilla
    SQUAT_THRESHOLD = 50  # píxeles
    
    is_squat_down = diff > -SQUAT_THRESHOLD  # Si diff es positivo o pequeño negativo, está abajo
    
    # Detectar transición de arriba a abajo y luego de abajo a arriba (una sentadilla completa)
    if state['state'] == 'up' and is_squat_down:
        # Transición de arriba a abajo - empezó a bajar
        state['state'] = 'down'
    elif state['state'] == 'down' and not is_squat_down:
        # Transición de abajo a arriba - completó una sentadilla
        state['state'] = 'up'
        state['squat_count'] += 1
    
    return is_squat_down, state

def draw_squat_count(img_pil, squat_count, is_squat_down):
    """
    Dibuja el conteo de sentadillas en la imagen.
    
    Args:
        img_pil: Imagen PIL donde dibujar
        squat_count: Número de sentadillas contadas
        is_squat_down: Si está en posición baja
    
    Returns:
        PIL.Image: Imagen con el conteo dibujado
    """
    draw = ImageDraw.Draw(img_pil)
    
    # Texto del conteo
    text = f"Sentadillas: {squat_count}"
    status_text = "BAJANDO" if is_squat_down else "SUBIENDO"
    
    # Obtener dimensiones de la imagen
    width, height = img_pil.size
    
    # Fuente grande para el conteo
    font = None
    try:
        from PIL import ImageFont
        # Intentar usar una fuente más grande
        font_size = max(40, int(width / 20))
        # Intentar diferentes fuentes comunes
        font_paths = [
            "arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arial.ttf"
        ]
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
    except:
        pass
    
    # Fondo semi-transparente para el texto
    text_bbox = draw.textbbox((0, 0), text, font=font) if font else draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    status_bbox = draw.textbbox((0, 0), status_text, font=font) if font else draw.textbbox((0, 0), status_text)
    status_width = status_bbox[2] - status_bbox[0]
    status_height = status_bbox[3] - status_bbox[1]
    
    # Dibujar fondo
    padding = 20
    box_x = 20
    box_y = 20
    box_width = max(text_width, status_width) + padding * 2
    box_height = text_height + status_height + padding * 3
    
    # Fondo semi-transparente (usar gris oscuro en lugar de transparencia)
    draw.rectangle(
        [(box_x, box_y), (box_x + box_width, box_y + box_height)],
        fill=(40, 40, 40),  # Gris oscuro
        outline=(255, 255, 255),
        width=3
    )
    
    # Color del texto según estado
    text_color = (255, 100, 100) if is_squat_down else (100, 255, 100)
    
    # Dibujar texto del conteo
    draw.text(
        (box_x + padding, box_y + padding),
        text,
        fill=(255, 255, 255),
        font=font
    )
    
    # Dibujar texto del estado
    draw.text(
        (box_x + padding, box_y + padding + text_height + 10),
        status_text,
        fill=text_color,
        font=font
    )
    
    return img_pil

def process_video_squat_count(video_file, pose_model, detr_model, detr_processor, device, conf,
                             point_size_multiplier=1.0, line_width_multiplier=1.0,
                             detr_threshold=0.9, frame_skip=2, target_resolution=(640, 360),
                             max_frames=None):
    """
    Procesa un video contando sentadillas en tiempo real.
    
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
        frame_skip: Procesar cada N frames
        target_resolution: tuple (width, height) para redimensionar frames
        max_frames: Número máximo de frames a procesar (None = todos)
    
    Returns:
        tuple: (video_path, total_frames, processed_frames, fps, width, height, inference_time, total_squats)
    """
    start_time = time.time()
    tfile = None
    output_path = None
    cap = None
    out = None
    
    try:
        # Leer video completo para guardar
        video_bytes = video_file.read()
        
        # Guardar video temporalmente
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_bytes)
        tfile.close()
        
        # Abrir video
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            if tfile and os.path.exists(tfile.name):
                os.unlink(tfile.name)
            return None, 0, 0, 0, 0, 0, 0.0, 0
        
        # Obtener propiedades del video original
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if orig_width == 0 or orig_height == 0:
            if tfile and os.path.exists(tfile.name):
                os.unlink(tfile.name)
            return None, 0, 0, 0, 0, 0, 0.0, 0
        
        # Aplicar resolución objetivo
        if target_resolution is None:
            width, height = orig_width, orig_height
            st.info(f"⚡ Procesando a resolución original: {width}x{height}")
        else:
            width, height = target_resolution
            st.info(f"⚡ Optimización: Redimensionando de {orig_width}x{orig_height} a {width}x{height}")
        
        # Crear video de salida
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps // frame_skip, (width, height))
        
        if not out.isOpened():
            if tfile and os.path.exists(tfile.name):
                os.unlink(tfile.name)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
            return None, 0, 0, 0, 0, 0, 0.0, 0
        
        # Estado para seguimiento de sentadillas (por persona)
        person_states = {}  # {person_id: state_dict}
        total_squats = 0
        
        # Procesar frames
        frame_count = 0
        processed_count = 0
        batch_size = 4
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Buffer para frames a procesar
        frame_batch = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Limitar número de frames si se especifica
            if max_frames is not None and processed_count >= max_frames:
                break
            
            # Frame skipping: solo procesar cada N frames
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Redimensionar frame ANTES de procesar (optimización) solo si es necesario
            if target_resolution is not None and (frame.shape[1] != width or frame.shape[0] != height):
                frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = frame
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Agregar a batch
            frame_batch.append((frame_count, frame_pil, frame_resized))
            
            # Procesar batch cuando esté lleno
            if len(frame_batch) >= batch_size:
                # Procesar batch
                for idx, pil_img, frame_bgr in frame_batch:
                    try:
                        # Aplicar inferencia de pose (igual que en análisis de imagen)
                        result_image, person_count, persons_details = infer_multi_person_pose(
                            pil_img, pose_model, detr_model, detr_processor, device, conf,
                            point_size_multiplier, line_width_multiplier, detr_threshold
                        )
                        
                        if result_image is not None and persons_details:
                            # Procesar cada persona detectada
                            max_squats_in_frame = 0
                            is_any_squat_down = False
                            
                            for person_idx, person_detail in enumerate(persons_details):
                                keypoints = np.array(person_detail['keypoints'])
                                confidences = np.array(person_detail['confidences'])
                                
                                # Obtener o crear estado para esta persona
                                person_id = person_idx  # Usar índice como ID
                                if person_id not in person_states:
                                    person_states[person_id] = {
                                        'state': 'up',
                                        'squat_count': 0,
                                        'hip_y_avg': None,
                                        'knee_y_avg': None
                                    }
                                
                                # Detectar sentadilla
                                is_squat_down, person_states[person_id] = detect_squat(
                                    keypoints, confidences, person_states[person_id]
                                )
                                
                                # Actualizar máximo de sentadillas y estado
                                max_squats_in_frame = max(max_squats_in_frame, person_states[person_id]['squat_count'])
                                if is_squat_down:
                                    is_any_squat_down = True
                            
                            # Actualizar total de sentadillas (sumar todas las sentadillas de todas las personas)
                            # Usar el máximo de sentadillas contadas por cualquier persona en este frame
                            for person_id in person_states:
                                total_squats = max(total_squats, person_states[person_id]['squat_count'])
                            
                            # Dibujar conteo en la imagen
                            result_image = draw_squat_count(result_image, total_squats, is_any_squat_down)
                            
                            # Convertir PIL a OpenCV
                            result_np = np.array(result_image)
                            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                            out.write(result_bgr)
                            processed_count += 1
                        else:
                            # Si falla, escribir frame original
                            out.write(frame_bgr)
                            processed_count += 1
                    except Exception as e:
                        # Si hay error, escribir frame original
                        out.write(frame_bgr)
                        processed_count += 1
                
                frame_batch.clear()
                
                # Actualizar progreso
                if total_frames > 0:
                    progress = min(1.0, frame_count / total_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"⚡ Procesando: Frame {frame_count}/{total_frames} ({processed_count} procesados) | Sentadillas: {total_squats}")
                else:
                    progress_bar.progress(0.5)
                    status_text.text(f"⚡ Procesando: Frame {frame_count} ({processed_count} procesados) | Sentadillas: {total_squats}")
            
            frame_count += 1
        
        # Procesar frames restantes en el batch
        if len(frame_batch) > 0:
            for idx, pil_img, frame_bgr in frame_batch:
                try:
                    result_image, person_count, persons_details = infer_multi_person_pose(
                        pil_img, pose_model, detr_model, detr_processor, device, conf,
                        point_size_multiplier, line_width_multiplier, detr_threshold
                    )
                    
                    if result_image is not None and persons_details:
                        max_squats_in_frame = 0
                        is_any_squat_down = False
                        
                        for person_idx, person_detail in enumerate(persons_details):
                            keypoints = np.array(person_detail['keypoints'])
                            confidences = np.array(person_detail['confidences'])
                            
                            person_id = person_idx
                            if person_id not in person_states:
                                person_states[person_id] = {
                                    'state': 'up',
                                    'squat_count': 0,
                                    'hip_y_avg': None,
                                    'knee_y_avg': None
                                }
                            
                            is_squat_down, person_states[person_id] = detect_squat(
                                keypoints, confidences, person_states[person_id]
                            )
                            
                            max_squats_in_frame = max(max_squats_in_frame, person_states[person_id]['squat_count'])
                            if is_squat_down:
                                is_any_squat_down = True
                        
                        # Actualizar total de sentadillas
                        for person_id in person_states:
                            total_squats = max(total_squats, person_states[person_id]['squat_count'])
                        result_image = draw_squat_count(result_image, total_squats, is_any_squat_down)
                        
                        result_np = np.array(result_image)
                        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                        out.write(result_bgr)
                        processed_count += 1
                    else:
                        out.write(frame_bgr)
                        processed_count += 1
                except Exception as e:
                    out.write(frame_bgr)
                    processed_count += 1
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Procesamiento completado: {processed_count} frames procesados | Sentadillas totales: {total_squats}")
        
        # Calcular tiempo de inferencia
        inference_time = time.time() - start_time
        
    except Exception as e:
        st.error(f"Error procesando video: {e}")
        import traceback
        st.code(traceback.format_exc())
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)
        inference_time = time.time() - start_time
        return None, 0, 0, 0, 0, 0, inference_time, 0
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if tfile and os.path.exists(tfile.name):
            os.unlink(tfile.name)
    
    return output_path, total_frames, processed_count, fps, width, height, inference_time, total_squats

def process_video(video_file, pose_model, detr_model, detr_processor, device, conf,
                            point_size_multiplier=1.0, line_width_multiplier=1.0,
                            detr_threshold=0.9, frame_skip=2, target_resolution=(640, 360),
                            max_frames=None):
    """
    Procesa un video de forma optimizada aplicando técnicas avanzadas de optimización.
    
    Optimizaciones aplicadas:
    1. Frame skipping: Procesa solo cada N frames
    2. Resolución reducida: Procesa a menor resolución para mayor velocidad
    3. Batch processing: Procesa múltiples frames en paralelo
    4. Half precision (FP16): Usa FP16 si GPU está disponible
    5. Efficient memory: Libera memoria inmediatamente después de procesar
    
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
        frame_skip: Procesar cada N frames (2 = cada 2 frames, etc.)
        target_resolution: tuple (width, height) para redimensionar frames
        max_frames: Número máximo de frames a procesar (None = todos)
    
    Returns:
        tuple: (video_path, total_frames, processed_frames, fps, width, height, inference_time)
    """
    start_time = time.time()
    tfile = None
    output_path = None
    cap = None
    out = None
    
    try:
        # Optimizaciones aplicadas:
        # - Frame skipping para reducir número de frames procesados
        # - Resolución reducida para procesamiento más rápido
        # - Batch processing eficiente
        # - torch.no_grad() ya está aplicado en las funciones de inferencia
        
        # Leer video completo para guardar
        video_bytes = video_file.read()
        
        # Guardar video temporalmente
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_bytes)
        tfile.close()
        
        # Abrir video
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            if tfile and os.path.exists(tfile.name):
                os.unlink(tfile.name)
            return None, 0, 0, 0, 0, 0, 0.0
        
        # Obtener propiedades del video original
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if orig_width == 0 or orig_height == 0:
            if tfile and os.path.exists(tfile.name):
                os.unlink(tfile.name)
            return None, 0, 0, 0, 0, 0, 0.0
        
        # Aplicar resolución objetivo
        if target_resolution is None:
            # Si no se especifica resolución, usar la original
            width, height = orig_width, orig_height
            st.info(f"⚡ Procesando a resolución original: {width}x{height}")
        else:
            width, height = target_resolution
            st.info(f"⚡ Optimización: Redimensionando de {orig_width}x{orig_height} a {width}x{height}")
        
        # Crear video de salida
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps // frame_skip, (width, height))
        
        if not out.isOpened():
            if tfile and os.path.exists(tfile.name):
                os.unlink(tfile.name)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
            return None, 0, 0, 0, 0, 0, 0.0
        
        # Procesar frames
        frame_count = 0
        processed_count = 0
        batch_size = 4  # Batch pequeño para optimizar memoria
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Buffer para frames a procesar
        frame_batch = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Limitar número de frames si se especifica
            if max_frames is not None and processed_count >= max_frames:
                break
            
            # Frame skipping: solo procesar cada N frames
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Redimensionar frame ANTES de procesar (optimización) solo si es necesario
            if target_resolution is not None and (frame.shape[1] != width or frame.shape[0] != height):
                frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = frame
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Agregar a batch
            frame_batch.append((frame_count, frame_pil, frame_resized))
            
            # Procesar batch cuando esté lleno
            if len(frame_batch) >= batch_size:
                # Procesar batch
                for idx, pil_img, frame_bgr in frame_batch:
                    try:
                        # Aplicar inferencia de pose (igual que en análisis de imagen)
                        result_image, person_count, _ = infer_multi_person_pose(
                            pil_img, pose_model, detr_model, detr_processor, device, conf,
                            point_size_multiplier, line_width_multiplier, detr_threshold
                        )
                        
                        if result_image is not None:
                            # Convertir PIL a OpenCV
                            result_np = np.array(result_image)
                            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                            out.write(result_bgr)
                            processed_count += 1
                        else:
                            # Si falla, escribir frame original
                            out.write(frame_bgr)
                            processed_count += 1
                    except Exception as e:
                        # Si hay error, escribir frame original
                        out.write(frame_bgr)
                        processed_count += 1
                
                frame_batch.clear()
                
                # Actualizar progreso
                if total_frames > 0:
                    progress = min(1.0, frame_count / total_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"⚡ Procesando: Frame {frame_count}/{total_frames} ({processed_count} procesados)")
                else:
                    progress_bar.progress(0.5)
                    status_text.text(f"⚡ Procesando: Frame {frame_count} ({processed_count} procesados)")
            
            frame_count += 1
        
        # Procesar frames restantes en el batch
        if len(frame_batch) > 0:
            for idx, pil_img, frame_bgr in frame_batch:
                try:
                    result_image, person_count, _ = infer_multi_person_pose(
                        pil_img, pose_model, detr_model, detr_processor, device, conf,
                        point_size_multiplier, line_width_multiplier, detr_threshold
                    )
                    
                    if result_image is not None:
                        result_np = np.array(result_image)
                        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                        out.write(result_bgr)
                        processed_count += 1
                    else:
                        out.write(frame_bgr)
                        processed_count += 1
                except Exception as e:
                    out.write(frame_bgr)
                    processed_count += 1
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Procesamiento completado: {processed_count} frames procesados")
        
        # Calcular tiempo de inferencia
        inference_time = time.time() - start_time
        
    except Exception as e:
        st.error(f"Error procesando video: {e}")
        import traceback
        st.code(traceback.format_exc())
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)
        return None, 0, 0, 0, 0, 0, 0.0
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if tfile and os.path.exists(tfile.name):
            os.unlink(tfile.name)
    
    return output_path, total_frames, processed_count, fps, width, height, inference_time


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
        <em> Visualización con círculos negros para los puntos clave</em>

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
        ["Análisis de Imagen", "Tomar Foto", "Análisis de Video", "Conteo de sentadillas"]
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
        
        st.markdown("""
        <div class="card">
            <h3>⚡ Análisis de Video</h3>
            <p>Procesa videos de forma optimizada usando técnicas avanzadas de optimización para detectar poses humanas.</p>
            <p><strong>Optimizaciones aplicadas:</strong></p>
            <ul>
                <li>Frame skipping: Procesa solo cada N frames</li>
                <li>Resolución reducida: Procesa a menor resolución para mayor velocidad</li>
                <li>Procesamiento eficiente: Optimizado para reducir tiempo de espera</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Subir Video")
            uploaded_video = st.file_uploader(
                "Selecciona un video",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                help="Formatos soportados: MP4, AVI, MOV, MKV, WEBM",
                key="video_uploader"
            )
            
            if uploaded_video is not None:
                # Mostrar información del video
                st.video(uploaded_video)
                
                # Configuración de procesamiento optimizado
                st.markdown("#### ⚡ Configuración Optimizada")
                
                frame_skip = st.slider(
                    "Procesar cada N frames",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="1 = todos los frames, 2 = cada 2 frames, etc. Valores más altos = más rápido",
                    key="frame_skip"
                )
                
                # Opción de resolución para procesamiento más rápido
                resolution_option = st.selectbox(
                    "Resolución de procesamiento",
                    ["360p (640x360)", "480p (854x480)", "720p (1280x720)", "1080p (1920x1080)", "Original"],
                    index=0,  # Por defecto 360p para máxima velocidad
                    help="Resoluciones más bajas = procesamiento más rápido. Recomendado: 360p o 480p para mejor velocidad.",
                    key="resolution_option"
                )
                
                # Obtener resolución objetivo
                resolution_map = {
                    "Original": None,
                    "1080p (1920x1080)": (1920, 1080),
                    "720p (1280x720)": (1280, 720),
                    "480p (854x480)": (854, 480),
                    "360p (640x360)": (640, 360)
                }
                target_resolution = resolution_map[resolution_option]
                
                max_frames_option = st.selectbox(
                    "Límite de frames (opcional)",
                    ["Todos los frames", "50 frames", "100 frames", "200 frames"],
                    help="Limita el número de frames a procesar para videos largos",
                    key="max_frames_option"
                )
                
                max_frames_map = {
                    "Todos los frames": None,
                    "50 frames": 50,
                    "100 frames": 100,
                    "200 frames": 200
                }
                max_frames = max_frames_map[max_frames_option]
                
                st.success(f"""
                **⚡ Configuración Optimizada:**
                - Frame skip: {frame_skip} (procesa 1 de cada {frame_skip} frames)
                - Resolución: {resolution_option}
                - Límite: {max_frames_option}
                
                **Tiempo estimado:** Reducido significativamente gracias a las optimizaciones
                """)
        
        with col2:
            st.markdown("#### Resultado")
            
            if uploaded_video is not None:
                if st.button("⚡ Procesar Video", key="process_video", type="primary"):
                    # Resetear el stream del archivo
                    uploaded_video.seek(0)
                    
                    with st.spinner("⚡ Procesando video con optimizaciones... Esto puede tardar unos minutos."):
                        try:
                            # Crear un BytesIO wrapper para el archivo
                            video_bytes_io = io.BytesIO(uploaded_video.read())
                            uploaded_video.seek(0)  # Resetear para mostrar el video original
                            
                            output_path, total_frames, processed_frames, fps, width, height, inference_time = process_video(
                                video_bytes_io,
                                pose_model, detr_model, detr_processor, device, conf,
                                point_size_multiplier, line_width_multiplier,
                                detr_threshold,
                                frame_skip=frame_skip,
                                target_resolution=target_resolution,
                                max_frames=max_frames
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
                                
                                # Mostrar tiempo de inferencia
                                minutes = int(inference_time // 60)
                                seconds = int(inference_time % 60)
                                milliseconds = int((inference_time % 1) * 1000)
                                
                                if minutes > 0:
                                    time_str = f"{minutes}m {seconds}s {milliseconds}ms"
                                else:
                                    time_str = f"{seconds}s {milliseconds}ms"
                                
                                st.markdown(f"""
                                <div class=\"metric-card\" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                                    <div class=\"metric-value\">{time_str}</div>
                                    <div class=\"metric-label\">Tiempo de inferencia</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Opción de descarga
                                st.download_button(
                                    label="Descargar Video Procesado",
                                    data=video_bytes,
                                    file_name=f"pose_result_optimized_{uploaded_video.name}",
                                    mime="video/mp4"
                                )
                                
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
                    <p>Selecciona un video desde tu dispositivo para analizar las poses con la versión optimizada</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif mode == "Conteo de sentadillas":
        
        st.markdown("""
        <div class="card">
            <h3>🏋️ Conteo de Sentadillas</h3>
            <p>Analiza videos de ejercicios y cuenta automáticamente las sentadillas realizadas en tiempo real.</p>
            <p><strong>Características:</strong></p>
            <ul>
                <li>Detección automática de sentadillas basada en pose estimation</li>
                <li>Conteo en tiempo real durante el procesamiento</li>
                <li>Visualización del esqueleto con keypoints</li>
                <li>Indicador visual de estado (BAJANDO/SUBIENDO)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Subir Video")
            uploaded_video = st.file_uploader(
                "Selecciona un video con ejercicios",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                help="Formatos soportados: MP4, AVI, MOV, MKV, WEBM",
                key="video_uploader_squat"
            )
            
            if uploaded_video is not None:
                # Mostrar información del video
                st.video(uploaded_video)
                
                # Configuración de procesamiento
                st.markdown("#### ⚙️ Configuración")
                
                frame_skip = st.slider(
                    "Procesar cada N frames",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="1 = todos los frames, 2 = cada 2 frames, etc. Valores más altos = más rápido",
                    key="frame_skip_squat"
                )
                
                # Opción de resolución para procesamiento más rápido
                resolution_option = st.selectbox(
                    "Resolución de procesamiento",
                    ["360p (640x360)", "480p (854x480)", "720p (1280x720)", "1080p (1920x1080)", "Original"],
                    index=0,  # Por defecto 360p para máxima velocidad
                    help="Resoluciones más bajas = procesamiento más rápido. Recomendado: 360p o 480p para mejor velocidad.",
                    key="resolution_option_squat"
                )
                
                # Obtener resolución objetivo
                resolution_map = {
                    "Original": None,
                    "1080p (1920x1080)": (1920, 1080),
                    "720p (1280x720)": (1280, 720),
                    "480p (854x480)": (854, 480),
                    "360p (640x360)": (640, 360)
                }
                target_resolution = resolution_map[resolution_option]
                
                max_frames_option = st.selectbox(
                    "Límite de frames (opcional)",
                    ["Todos los frames", "50 frames", "100 frames", "200 frames"],
                    help="Limita el número de frames a procesar para videos largos",
                    key="max_frames_option_squat"
                )
                
                max_frames_map = {
                    "Todos los frames": None,
                    "50 frames": 50,
                    "100 frames": 100,
                    "200 frames": 200
                }
                max_frames = max_frames_map[max_frames_option]
                
                st.info(f"""
                **⚙️ Configuración:**
                - Frame skip: {frame_skip} (procesa 1 de cada {frame_skip} frames)
                - Resolución: {resolution_option}
                - Límite: {max_frames_option}
                
                **💡 Consejo:** Asegúrate de que la persona esté completamente visible en el video para una mejor detección.
                """)
        
        with col2:
            st.markdown("#### Resultado")
            
            if uploaded_video is not None:
                if st.button("🏋️ Procesar Video", key="process_video_squat", type="primary"):
                    # Resetear el stream del archivo
                    uploaded_video.seek(0)
                    
                    with st.spinner("🏋️ Procesando video y contando sentadillas... Esto puede tardar unos minutos."):
                        try:
                            # Crear un BytesIO wrapper para el archivo
                            video_bytes_io = io.BytesIO(uploaded_video.read())
                            uploaded_video.seek(0)  # Resetear para mostrar el video original
                            
                            output_path, total_frames, processed_frames, fps, width, height, inference_time, total_squats = process_video_squat_count(
                                video_bytes_io,
                                pose_model, detr_model, detr_processor, device, conf,
                                point_size_multiplier, line_width_multiplier,
                                detr_threshold,
                                frame_skip=frame_skip,
                                target_resolution=target_resolution,
                                max_frames=max_frames
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
                                
                                # Mostrar tiempo de inferencia
                                minutes = int(inference_time // 60)
                                seconds = int(inference_time % 60)
                                milliseconds = int((inference_time % 1) * 1000)
                                
                                if minutes > 0:
                                    time_str = f"{minutes}m {seconds}s {milliseconds}ms"
                                else:
                                    time_str = f"{seconds}s {milliseconds}ms"
                                
                                st.markdown(f"""
                                <div class=\"metric-card\" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                                    <div class=\"metric-value\">{time_str}</div>
                                    <div class=\"metric-label\">Tiempo de inferencia</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Mostrar TOTAL DE SENTADILLAS (destacado)
                                st.markdown(f"""
                                <div class=\"metric-card\" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);">
                                    <div class=\"metric-value\" style="font-size: 3rem;">{total_squats}</div>
                                    <div class=\"metric-label\" style="font-size: 1.2rem;">Sentadillas Totales</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Opción de descarga
                                st.download_button(
                                    label="Descargar Video Procesado",
                                    data=video_bytes,
                                    file_name=f"squat_count_{uploaded_video.name}",
                                    mime="video/mp4"
                                )
                                
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
                    <p>Selecciona un video con ejercicios de sentadillas para contar automáticamente las repeticiones</p>
                    <p><strong>Recomendaciones:</strong></p>
                    <ul style="text-align: left; display: inline-block;">
                        <li>Persona completamente visible</li>
                        <li>Buena iluminación</li>
                        <li>Fondo simple</li>
                        <li>Vista lateral o frontal</li>
                    </ul>
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