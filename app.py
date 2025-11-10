import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import os
import io
import base64
import cv2
import time
from transformers import DetrImageProcessor, DetrForObjectDetection

"""
CONFIGURACIÓN DE MODELOS:

• Pose Transformer: HRNet-W32, entrada 512x384, entrenado en COCO
• DETR: ResNet-101, detector de objetos general (usado para detectar personas)
• Dispositivo: Usa GPU si está disponible, sino CPU

NOTA: Los modelos se cargan una vez y se cachean (@st.cache_resource) para mejor rendimiento.
Esto evita recargarlos en cada interacción del usuario, mejorando significativamente la velocidad.
"""

# Configurar la página
st.set_page_config(
    page_title="Pose Estimation App",
    page_icon="🤸",
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

# Instalar soporte para AVIF si está disponible
try:
    import pillow_avif
    st.success(" Soporte AVIF disponible")
except ImportError:
    st.warning(" Soporte AVIF no disponible - las imágenes AVIF no se podrán procesar")

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
    try:
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
    except Exception as e:
        st.error(f"Error cargando los modelos: {e}")
        return None, None, None, None, None

# ---------- Normalización ----------
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Variables globales para el modelo de pose
model_w, model_h = 512, 384  # Tamaño por defecto del modelo

# ---------- Esqueleto para dibujar - COCO ----------
# Skeleton EXACTO de la clase Visualizer (1-based)
SKELETON_1BASED = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
                   [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                   [2, 4], [3, 5], [4, 6], [5, 7]]
SKELETON = [[a-1, b-1] for a, b in SKELETON_1BASED]  # Convertir a 0-based

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
DISEÑO VISUAL - Basado en Visualizer original:

• Esqueleto: Líneas gruesas coloridas por segmento corporal
  - Verde: brazo izquierdo (hombro → codo → muñeca)
  - Amarillo: brazo derecho (hombro → codo → muñeca)
  - Azul: pierna izquierda (hombro → cadera → rodilla → tobillo)
  - Rosa: pierna derecha (hombro → cadera → rodilla → tobillo)
  - Rosa claro: conexiones de cabeza (líneas delgadas)
  
• Keypoints: Círculos negros con tamaño proporcional
  - Cabeza: círculos más pequeños (1.5x escala)
  - Cuerpo: círculos más grandes (3.0x escala)
  
• Visibilidad: Basada en confianza > 80% (configurable)
• Escala: Ajusta automáticamente al tamaño de la imagen
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
    
    Nota: Solo dibuja keypoints con confianza > 80%. Los keypoints con coordenadas NaN
    o confianza baja no se visualizan.
    """
    draw = ImageDraw.Draw(img_pil)
    
    # Colores EXACTOS de la clase Visualizer (0-based) - COMPLETOS
    GREEN = [(4,5),(5,7),(7,9)]           # left_shoulder->left_elbow->left_wrist
    YELLOW = [(4,6),(6,8),(8,10)]         # right_shoulder->right_elbow->right_wrist
    BLUE = [(5,11),(11,13),(13,15)]       # left_shoulder->left_hip->left_knee->left_ankle
    PINK = [(6,12),(12,14),(14,16)]       # right_shoulder->right_hip->right_knee->right_ankle
    
    # Convertir confidences a vis mask
    vis = np.ones((17,), dtype=np.int32)  # Por defecto todos visibles
    if confidences is not None:
        confidences = np.asarray(confidences)
        vis = (confidences > 0.8).astype(np.int32)
    
    # Dibujar líneas del skeleton EXACTAMENTE como Visualizer
    for i, j in SKELETON_1BASED:  # Usar 1-based
        i_idx, j_idx = i - 1, j - 1  # Convertir a 0-based para índices
        if vis[i_idx] <= 0 or vis[j_idx] <= 0:
            continue
        
        src = keypoints[i_idx]
        dst = keypoints[j_idx]
        
        # Calcular ki, kj como en Visualizer
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

def draw_keypoints_on_cv2(img_cv2, keypoints, radius=3, line_width=2):
    """Dibujar keypoints en imagen OpenCV"""
    img_copy = img_cv2.copy()
    for a,b in SKELETON:
        a_idx, b_idx = a, b  # Ya son 0-indexed
        if np.any(np.isnan(keypoints[a_idx])) or np.any(np.isnan(keypoints[b_idx])):
            continue
        pt1 = (int(keypoints[a_idx][0]), int(keypoints[a_idx][1]))
        pt2 = (int(keypoints[b_idx][0]), int(keypoints[b_idx][1]))
        cv2.line(img_copy, pt1, pt2, (0, 255, 0), line_width)
    
    for (x,y) in keypoints:
        if np.isnan(x) or np.isnan(y):
            continue
        center = (int(x), int(y))
        cv2.circle(img_copy, center, radius, (255, 0, 0), -1)
    
    return img_copy

def cv2_to_pil(cv2_img):
    """Convertir imagen OpenCV a PIL"""
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def pil_to_cv2(pil_img):
    """Convertir imagen PIL a OpenCV"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

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
    try:
        img_size = conf.MODEL.IMAGE_SIZE
        model_w, model_h = int(img_size[0]), int(img_size[1])
    except Exception:
        model_w, model_h = 512, 384
    # Obtener dimensiones del recorte
    crop_w, crop_h = person_crop_pil.size
    x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded = bbox_expanded
    
    # Crear metadatos de centro y escala a partir del tamaño del recorte
    center, scale = make_meta_from_wh(crop_w, crop_h, use_max_scale=True)
    
    meta = {
        'center': center,
        'scale': scale,
        'rotation': 0,
        'joints_3d': np.zeros((17, 3), dtype=np.float32),
        'joints_3d_vis': np.ones((17, 3), dtype=np.float32)
    }
    
    # Preprocesar con transformación afín al tamaño esperado por el modelo
    input_patch = preprocess_patch(person_crop_pil, center, scale, (model_w, model_h))
    
    # Forward pass original
    input_tensor = normalize(Image.fromarray(input_patch)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = pose_model(input_tensor)
    
    # Forward pass flipped
    input_flipped = np.flip(input_patch, axis=1).copy()
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
    
    # DEBUG: Verificar pipeline de transformación
    print(f"TRANSFORMACIÓN - Recorte: {crop_w}x{crop_h}, Bbox: {bbox_expanded}")
    print(f"TRANSFORMACIÓN - Centro: {center}, Escala: {scale}")
    
    # Aplicar filtrado
    conf_th = 0.8
    valid_mask = np.zeros((17,), dtype=np.int32)
    
    if isinstance(outputs, dict) and 'pred_logits' in outputs and 'pred_coords' in outputs:
        logits = outputs['pred_logits']
        pred_coords_q = outputs['pred_coords']
        
        logits_f = outputs_flipped['pred_logits']
        pred_coords_q_f = outputs_flipped['pred_coords']
        
        probs = torch.softmax((logits + logits_f) / 2.0, dim=-1)[0].cpu().numpy()
        coords_q = ((pred_coords_q + pred_coords_q_f) / 2.0)[0].cpu().numpy()
        
        if coords_q.max() <= 1.01:
            coords_q[:, 0] *= model_w
            coords_q[:, 1] *= model_h
        
        # HUNGARIAN MATCHING - Alineación queries↔keypoints
        # El modelo PRTR usa 100 queries, pero solo necesitamos 17 keypoints
        # Buscamos la mejor query para cada keypoint basado en:
        # 1. Probabilidad de clase > 30% (ACCEPT_PROB_TH)
        # 2. Distancia espacial < 50 píxeles (ACCEPT_DIST_TH)
        # 3. Confianza del heatmap > 80% (conf_th)
        # Si no se cumple pero hay queries de soporte cercanas, se acepta igualmente
        for j in range(17):
            class_probs = probs[:, j]
            best_query_idx = np.argmax(class_probs)
            best_prob = class_probs[best_query_idx]
            query_coord = coords_q[best_query_idx]
            joint_coord = preds_final[j]
            distance = np.linalg.norm(query_coord - joint_coord)
            
            ACCEPT_PROB_TH = 0.3
            ACCEPT_DIST_TH = 50.0
            
            accepted = (best_prob > ACCEPT_PROB_TH and 
                       distance < ACCEPT_DIST_TH and 
                       heat_conf[j] > conf_th)
            
            # Si no se cumplen las condiciones estrictas pero hay queries de soporte,
            # aceptamos el keypoint (heurística para casos ambiguos)
            if not accepted and heat_conf[j] > conf_th:
                supporting_queries = np.where((class_probs > 0.1) & 
                                            (np.linalg.norm(coords_q - joint_coord, axis=1) < 80.0))[0]
                if len(supporting_queries) > 0:
                    accepted = True
            
            valid_mask[j] = 1 if accepted else 0
    else:
        valid_mask = (heat_conf > conf_th).astype(np.int32)
    
    # Filtrar keypoints
    filtered_preds = preds_final.copy()
    filtered_scores = heat_conf.copy()
    
    for i in range(17):
        if valid_mask[i] == 0:
            filtered_preds[i] = [np.nan, np.nan]
            filtered_scores[i] = 0.0
    
    # CORRECCIÓN CRÍTICA: Transformar coordenadas del espacio del modelo al espacio del recorte
    # y luego al espacio de la imagen original
    
    # Las coordenadas de get_final_preds_match están en el espacio del recorte transformado
    # Necesitamos mapearlas al espacio del recorte original primero
    
    # 1. Crear transformación inversa para mapear del espacio del modelo al espacio del recorte original
    trans_inv = get_affine_transform(center, scale, 0, (model_w, model_h), inv=1)
    
    # 2. Transformar cada keypoint al espacio del recorte original
    keypoints_crop_space = []
    for kp in filtered_preds:
        if not np.isnan(kp[0]):
            # Aplicar transformación inversa
            kp_homogeneous = np.array([kp[0], kp[1], 1.0])
            kp_original = trans_inv.dot(kp_homogeneous)
            keypoints_crop_space.append([kp_original[0], kp_original[1]])
        else:
            keypoints_crop_space.append([np.nan, np.nan])
    
    keypoints_crop_space = np.array(keypoints_crop_space)
    
    # 3. Transformar del espacio del recorte al espacio de la imagen original
    keypoints_original = keypoints_crop_space.copy()
    keypoints_original[:, 0] += x_min_expanded
    keypoints_original[:, 1] += y_min_expanded
    
    # DEBUG: Verificar pipeline de transformación
    print(f"TRANSFORMACIÓN - Keypoint 0: modelo[{filtered_preds[0]}] → recorte[{keypoints_crop_space[0]}] → original[{keypoints_original[0]}]")
    
    # Calcular plot_scale
    plot_scale = np.linalg.norm(scale) / 2.0
    
    return keypoints_original, filtered_scores.reshape(-1, 1), plot_scale

def apply_nms_to_persons(persons, nms_threshold=0.5):
    """
    Aplicar Non-Maximum Suppression a las detecciones de personas para eliminar duplicados
    """
    if len(persons) <= 1:
        return persons
    
    # Convertir a formato numpy para cálculos
    scores = np.array([score.item() for score, _ in persons])
    boxes = np.array([[box[0].item(), box[1].item(), box[2].item(), box[3].item()] for _, box in persons])
    
    # Ordenar por score descendente
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Tomar el box con mayor score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        # Calcular IoU con el resto de boxes
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        # Calcular IoU
        ious = calculate_iou(current_box, other_boxes)
        
        # Mantener solo boxes con IoU menor al threshold
        indices = indices[1:][ious < nms_threshold]
    
    # Retornar solo las personas seleccionadas
    return [persons[i] for i in keep]

def calculate_iou(box1, boxes2):
    """
    Calcular Intersection over Union (IoU) entre un box y múltiples boxes
    """
    # box1: [x1, y1, x2, y2]
    # boxes2: [[x1, y1, x2, y2], ...]
    
    # Coordenadas de intersección
    x1 = np.maximum(box1[0], boxes2[:, 0])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y2 = np.minimum(box1[3], boxes2[:, 3])
    
    # Calcular área de intersección
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calcular área de cada box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calcular área de unión
    union = area1 + area2 - intersection
    
    # Calcular IoU
    iou = intersection / union
    
    return iou

def infer_multi_person_pose(image_pil, pose_model, detr_model, detr_processor, device, conf, 
                           point_size_multiplier=1.0, line_width_multiplier=1.0,
                           detr_threshold=0.9, nms_threshold=0.5):
    """
    Inferencia multi-persona usando DETR para detectar personas y luego keypoints.

    FLUJO MULTI-PERSONA:
    1. DETR detecta todas las personas en la imagen
    2. Aplicamos NMS para eliminar cajas duplicadas (DETR tiende a duplicar detecciones)
    3. Para cada persona única:
       - Expandimos la bbox 25% en cada dirección (da contexto extra para keypoints en bordes)
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
        nms_threshold (float): Umbral IoU para NMS (0.5 recomendado)

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
        # Usar threshold configurable para ser más selectivo y evitar duplicados
        results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=detr_threshold)[0]
        
        # Filtrar solo detecciones de personas (label == 1 en COCO dataset)
        person_count = 0
        
        # Primero, obtener todas las personas detectadas
        persons = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # En COCO dataset, label 1 es "person"
            if label.item() == 1:  # Solo personas
                persons.append((score, box))
        
        # Aplicar Non-Maximum Suppression (NMS) para eliminar detecciones duplicadas
        original_count = len(persons)
        if len(persons) > 1:
            persons = apply_nms_to_persons(persons, nms_threshold=nms_threshold)
            filtered_count = len(persons)
            if original_count != filtered_count:
                st.info(f"NMS aplicado: {original_count} -> {filtered_count} personas (eliminadas {original_count - filtered_count} duplicados)")
        
        # Crear una copia de la imagen original donde dibujaremos todos los keypoints
        # Crear imagen para dibujar keypoints
        image_with_keypoints = image_pil.copy()
        
        # Procesar cada persona detectada
        persons_details = []
        for idx, (score, box) in enumerate(persons, 1):
            person_count += 1
            
            # Convertir coordenadas: [x_min, y_min, x_max, y_max]
            box = [round(i, 2) for i in box.tolist()]
            
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            # Expandir la caja 50% (25% en cada lado)
            # La expansión de bbox ayuda con keypoints en bordes: dar contexto extra alrededor
            # de la persona mejora la detección de keypoints cerca de los bordes del recorte.
            EXPANSION_FACTOR = 0.25  # 25% en cada dirección = 50% total
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

                # Dibujar keypoints sobre la imagen original usando la nueva función
                # Usar plot_scale directamente como en test3.py con parámetros configurables
                image_with_keypoints = draw_keypoints_pil(image_with_keypoints, keypoints_absolute, 
                                                        confidences=keypoint_scores_crop.flatten(), 
                                                        scale=plot_scale,
                                                        point_size_multiplier=point_size_multiplier,
                                                        line_width_multiplier=line_width_multiplier)

                # Guardar detalles por persona
                persons_details.append({
                    'bbox': [x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded],
                    'keypoints': keypoints_absolute.tolist(),
                    'confidences': keypoint_scores_crop.flatten().tolist()
                })
                
            except Exception as e:
                st.warning(f"Error procesando keypoints para persona {person_count}: {e}")
        
        return image_with_keypoints, person_count, persons_details
        
    except Exception as e:
        st.error(f"Error en inferencia multi-persona: {e}")
        return None, 0

def infer_on_image_full(image_pil, model, device, conf, do_flip=True):
    """
    image_pil: PIL RGB original
    devuelve coords (K,2) en coordenadas de la imagen original
    """
    try:
        # Obtener tamaño del modelo
        try:
            img_size = conf.MODEL.IMAGE_SIZE
            model_w, model_h = int(img_size[0]), int(img_size[1])
        except Exception:
            # fallback: según tu mensaje previo el modelo espera 288 x 384 (w x h)
            model_w, model_h = 384, 384
        
        # resize al tamaño del modelo (PIL: (width, height))
        resized = image_pil.resize((model_w, model_h), Image.BILINEAR)

        # centro y scale en coordenadas originales (convención usada en muchos demos HRNet)
        orig_w, orig_h = image_pil.size
        c = np.array([orig_w * 0.5, orig_h * 0.5], dtype=np.float32)
        scale_val = max(orig_w, orig_h) / 200.0
        s = np.array([scale_val, scale_val], dtype=np.float32)

        # forward original (tensors en device)
        x = normalize(resized).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)

        # pasar salida al postprocess tal cual
        _, _, preds_raw = get_final_preds_match(conf, out, c, s)

        # preds_raw puede venir en distinto sistema de coordenadas.
        preds = preds_raw[0].copy()  # (K,2)

        # si las coordenadas están en el espacio del "resized" (es decir muy pequeñas y < model_w/model_h),
        # entonces las escalamos a la imagen original (interpolación lineal).
        # heurística: si la mayoría de puntos caben dentro de [0, model_w] x [0, model_h],
        # consideramos que están en coords de la imagen redimensionada.
        inside_resized = np.sum((preds[:, 0] >= 0) & (preds[:, 0] <= model_w) &
                                (preds[:, 1] >= 0) & (preds[:, 1] <= model_h))
        if inside_resized >= 0.5 * preds.shape[0]:
            scale_x = orig_w / float(model_w)
            scale_y = orig_h / float(model_h)
            preds[:, 0] = preds[:, 0] * scale_x
            preds[:, 1] = preds[:, 1] * scale_y
            return preds

        # si no estaban en coords de resized, asumimos que ya vienen mapeadas (por ejemplo get_final_preds_match ya devolvió coords originales)
        if do_flip:
            # predicción espejo y promedio (si hace falta)
            resized_flip = ImageOps.mirror(resized)
            x_f = normalize(resized_flip).unsqueeze(0).to(device)
            with torch.no_grad():
                out_f = model(x_f)
            _, _, preds_raw_f = get_final_preds_match(conf, out_f, c, s)
            preds_f = preds_raw_f[0].copy()

            # aplicar misma heurística al flip
            inside_resized_f = np.sum((preds_f[:, 0] >= 0) & (preds_f[:, 0] <= model_w) &
                                      (preds_f[:, 1] >= 0) & (preds_f[:, 1] <= model_h))
            if inside_resized_f >= 0.5 * preds_f.shape[0]:
                preds_f[:, 0] = preds_f[:, 0] * (orig_w / float(model_w))
                preds_f[:, 1] = preds_f[:, 1] * (orig_h / float(model_h))

            # promedio (ya ambas en coordenadas originales o ambas en resized escaladas)
            preds = (preds + preds_f) / 2.0

        return preds
    except Exception as e:
        st.error(f"Error en la inferencia: {e}")
        return None

def main():
    # Header principal con diseño mejorado
    st.markdown("""
    <div class="main-header">
        <h1>Multi-Person Pose Estimation App</h1>
        <p>Detección de poses de múltiples personas con DETR y Transformers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar con información y configuración
    with st.sidebar:
        st.markdown("### Información")
        st.markdown("""
        <div class="card">
        Esta aplicación utiliza modelos avanzados de IA para detectar y analizar múltiples personas en imágenes.
        
        **Características:**
        - Detección múltiple de personas con DETR
        - Detección de 17 keypoints corporales por persona
        - Visualización del esqueleto humano
        - Soporte para múltiples formatos de imagen
        - Captura de fotos con cámara
        - Análisis completo de poses en escenas multi-persona
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Configuración")
        st.info("La aplicación detecta automáticamente todas las personas en la imagen y extrae sus poses individuales.")
        
        st.markdown("### Detección de Personas")
        st.markdown("**Ajustes para evitar detecciones duplicadas:**")
        
        # Controles para DETR - CONTROL DE DETECCIONES DUPLICADAS
        # DETR threshold: Controla qué tan estricto es el filtro de confianza
        # • 0.9-0.95: Muy estricto - solo personas muy claras (recomendado)
        # • 0.7-0.8: Moderado - puede detectar duplicados
        # • < 0.6: Permisivo - muchas detecciones pero más ruido
        detr_threshold = st.slider(
            "Umbral de confianza DETR",
            min_value=0.1,
            max_value=0.99,
            value=0.9,
            step=0.05,
            help="""
            CONTROL DE DETECCIONES DUPLICADAS:
            • 0.9-0.95: Muy estricto - solo personas muy claras (recomendado)
            • 0.7-0.8: Moderado - puede detectar duplicados
            • < 0.6: Permisivo - muchas detecciones pero más ruido
            """
        )
        
        # NMS threshold: Controla qué tan agresivo es el filtro de duplicados
        # Valores más bajos = más agresivo eliminando solapamientos
        nms_threshold = st.slider(
            "Umbral NMS (eliminación duplicados)",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Umbral para eliminar detecciones duplicadas. Valores más bajos = más agresivo eliminando duplicados."
        )
        
        st.markdown(f"""
        <div class=\"card\">
        <strong>Configuración de detección:</strong><br>
        <strong>DETR threshold:</strong> {detr_threshold}<br>
        <strong>NMS threshold:</strong> {nms_threshold}<br><br>
        <small><em>Los cambios se aplicarán en la próxima imagen que analices</em></small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Ajustes Visuales")
        st.markdown("**Personaliza el tamaño de los elementos del esqueleto:**")
        
        # Contenedor para el slider de puntos
        st.markdown('<div class="slider-container">', unsafe_allow_html=True)
        point_size_multiplier = st.slider(
            "Tamaño de puntos (keypoints)",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Ajusta el tamaño de los puntos de los keypoints. Valores más altos = puntos más grandes."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Contenedor para el slider de líneas
        st.markdown('<div class="slider-container">', unsafe_allow_html=True)
        line_width_multiplier = st.slider(
            "Grosor de líneas (esqueleto)",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Ajusta el grosor de las líneas del esqueleto. Valores más altos = líneas más gruesas."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Mostrar valores actuales con mejor formato
        st.markdown(f"""
        <div class=\"card\">
        <strong>Configuración actual:</strong><br>
        <strong>Puntos:</strong> {point_size_multiplier}x<br>
        <strong>Líneas:</strong> {line_width_multiplier}x<br><br>
        <small><em>Los cambios se aplicarán en la próxima imagen que analices</em></small>
        </div>
        """, unsafe_allow_html=True)
        
    
    # Cargar modelos
    with st.spinner("Cargando modelos..."):
        pose_model, detr_model, detr_processor, device, conf = load_models()
    
    if pose_model is None or detr_model is None:
        st.error("No se pudieron cargar los modelos. Verifica que los archivos estén disponibles.")
        return
    
    st.markdown(f"""
    <div class="status-success">
        Modelos cargados exitosamente en {str(device).upper()}
    </div>
    """, unsafe_allow_html=True)
    
    # Crear pestañas usando radio buttons
    st.markdown("### Modo de Análisis")
    mode = st.radio(
        "Selecciona el modo de análisis:",
        ["Análisis de Imagen", "Tomar Foto"]
    )
    
    if mode == "Análisis de Imagen":
        st.markdown("### Análisis de Imagen")
        
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
                st.image(image, caption="Imagen Original", use_column_width=True)
                
                # Información de la imagen
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{image.size[0]} x {image.size[1]}</div>
                    <div class="metric-label">píxeles</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Resultado de la Predicción")
            
            if uploaded_file is not None:
                if st.button("Analizar Pose", key="analyze_image"):
                    with st.spinner("Procesando imagen..."):
                        # Realizar inferencia multi-persona
                        result_image, person_count, persons_details = infer_multi_person_pose(
                            image, pose_model, detr_model, detr_processor, device, conf,
                            point_size_multiplier, line_width_multiplier, detr_threshold, nms_threshold
                        )
                        
                        if result_image is not None:
                            # Mostrar resultado
                            st.image(result_image, caption="Pose Detection Result", use_column_width=True)
                            
                            # Estadísticas
                            st.markdown(f"""
                            <div class=\"metric-card\">
                                <div class=\"metric-value\">{person_count}</div>
                                <div class=\"metric-label\">personas detectadas</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Mostrar tabla de keypoints y confianzas por persona
                            if persons_details:
                                st.markdown("### Detalle de keypoints y confianzas")
                                for p_idx, details in enumerate(persons_details, start=1):
                                    with st.expander(f"Persona {p_idx} - bbox: [x1={details['bbox'][0]:.1f}, y1={details['bbox'][1]:.1f}, x2={details['bbox'][2]:.1f}, y2={details['bbox'][3]:.1f}]"):
                                        rows = []
                                        for k in range(min(17, len(details['keypoints']))):
                                            x, y = details['keypoints'][k]
                                            conf_score = details['confidences'][k] if k < len(details['confidences']) else 0.0
                                            rows.append({
                                                'keypoint': k + 1,
                                                'x': None if x is None else (float('nan') if (x != x) else round(float(x), 2)),
                                                'y': None if y is None else (float('nan') if (y != y) else round(float(y), 2)),
                                                'confidence': round(float(conf_score), 3)
                                            })
                                        st.table(rows)
                            
                            # Opción de descarga
                            buf = io.BytesIO()
                            result_image.save(buf, format="PNG")
                            byte_im = buf.getvalue()
                            
                            st.download_button(
                                label="Descargar Resultado",
                                data=byte_im,
                                file_name=f"pose_result_{uploaded_file.name}",
                                mime="image/png"
                            )
                        else:
                            st.error("No se pudo procesar la imagen")
            else:
                st.markdown("""
                <div class=\"camera-container\">
                    <h4>Sube una imagen para comenzar</h4>
                    <p>Selecciona una imagen desde tu dispositivo para analizar la pose</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif mode == "Tomar Foto":
        st.markdown("### Tomar Foto con Cámara")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Capturar Foto")
            
            # Widget de cámara de Streamlit
            camera_photo = st.camera_input("Toma una foto para analizar la pose")
            
            if camera_photo is not None:
                # Convertir la foto a PIL Image
                image = Image.open(camera_photo).convert("RGB")
                
                # Mostrar imagen capturada
                st.image(image, caption="Foto Capturada", use_column_width=True)
                
                # Información de la imagen
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{image.size[0]} x {image.size[1]}</div>
                    <div class="metric-label">píxeles</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Botón de análisis en la misma columna
                if st.button("Analizar Pose", key="analyze_camera_photo"):
                    with st.spinner("Procesando foto..."):
                        # Realizar inferencia multi-persona
                        result_image, person_count, persons_details = infer_multi_person_pose(
                            image, pose_model, detr_model, detr_processor, device, conf,
                            point_size_multiplier, line_width_multiplier, detr_threshold, nms_threshold
                        )
                        
                        if result_image is not None:
                            # Mostrar resultado
                            st.image(result_image, caption="Pose Detection Result", use_column_width=True)
                            
                            # Estadísticas
                            st.markdown(f"""
                            <div class=\"metric-card\">
                                <div class=\"metric-value\">{person_count}</div>
                                <div class=\"metric-label\">personas detectadas</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Mostrar tabla de keypoints y confianzas por persona
                            if persons_details:
                                st.markdown("### Detalle de keypoints y confianzas")
                                for p_idx, details in enumerate(persons_details, start=1):
                                    with st.expander(f"Persona {p_idx} - bbox: [x1={details['bbox'][0]:.1f}, y1={details['bbox'][1]:.1f}, x2={details['bbox'][2]:.1f}, y2={details['bbox'][3]:.1f}]"):
                                        rows = []
                                        for k in range(min(17, len(details['keypoints']))):
                                            x, y = details['keypoints'][k]
                                            conf_score = details['confidences'][k] if k < len(details['confidences']) else 0.0
                                            rows.append({
                                                'keypoint': k + 1,
                                                'x': None if x is None else (float('nan') if (x != x) else round(float(x), 2)),
                                                'y': None if y is None else (float('nan') if (y != y) else round(float(y), 2)),
                                                'confidence': round(float(conf_score), 3)
                                            })
                                        st.table(rows)
                            
                            # Opción de descarga
                            buf = io.BytesIO()
                            result_image.save(buf, format="PNG")
                            byte_im = buf.getvalue()
                            
                            st.download_button(
                                label="Descargar Resultado",
                                data=byte_im,
                                file_name=f"pose_result_camera_{int(time.time())}.png",
                                mime="image/png"
                            )
                        else:
                            st.error("No se pudo procesar la foto")
            else:
                st.markdown("""
                <div class=\"camera-container\">
                    <h4>Tomar Foto</h4>
                    <p>Usa el widget de cámara para tomar una foto y analizar la pose</p>
                    <p><strong>Características:</strong></p>
                    <ul>
                        <li>Captura instantánea con la cámara</li>
                        <li>Análisis inmediato de la pose</li>
                        <li>Descarga del resultado</li>
                        <li>Fácil de usar</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Instrucciones")
            st.markdown("""
            <div class=\"camera-container\">
                <h4>Cómo Usar la Cámara</h4>
                <p><strong>Pasos para analizar una pose:</strong></p>
                <ol>
                    <li>Haz clic en "Toma una foto"</li>
                    <li>Permite el acceso a la cámara</li>
                    <li>Posiciona la persona en el marco</li>
                    <li>Haz clic en "Tomar foto"</li>
                    <li>Presiona "Analizar Pose"</li>
                    <li>Descarga el resultado</li>
                </ol>
                <p><strong>Consejos:</strong></p>
                <ul>
                    <li>Buena iluminación mejora la detección</li>
                    <li>Persona completa en el marco</li>
                    <li>Fondo simple funciona mejor</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer mejorado
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h4>Desarrollado con Streamlit y PyTorch</h4>
        <p>Pose Estimation App - Detección de poses humanas con IA</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
