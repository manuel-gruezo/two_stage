from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
import numpy as np
import sys
import json
from pathlib import Path
import cv2

# Obtener el directorio raíz del proyecto (dos niveles arriba desde local/)
SCRIPT_DIR = Path(__file__).parent.resolve()  # local/
ROOT_DIR = SCRIPT_DIR.parent.resolve()  # directorio raíz

# Añadir el directorio raíz al path para importar models y lib
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'lib'))

import models

# Imports para el modelo de pose
from config import cfg as conf
from config import update_config
from utils.utils import model_key_helper
from core.inference import get_final_preds_match
from utils.transforms import get_affine_transform

# ---------- Configuración del modelo de pose ----------
class Args:
    cfg = str(ROOT_DIR / 'experiments/coco/transformer/w32_512x384_adamw_lr1e-4.yaml')
    opts = []
    modelDir = None
    logDir = None
    dataDir = None
    pretrained = str(ROOT_DIR / 'models/pytorch/pose_coco/pose_transformer_hrnet_w32_512x384.pth')

args = Args()
update_config(conf, args)

# Cargar modelo de pose
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device para modelo de pose: {device}")
pose_model = models.pose_transformer.get_pose_net(conf, is_train=False)
state = torch.load(args.pretrained, map_location='cpu')
pose_model.load_state_dict(model_key_helper(state), strict=False)
pose_model.to(device)
pose_model.eval()
print("Modelo de pose cargado")

# Cargar modelo DETR
local_model_path = str(ROOT_DIR / "models/detr-resnet-101")
detr_processor = DetrImageProcessor.from_pretrained(local_model_path)
detr_model = DetrForObjectDetection.from_pretrained(local_model_path)
detr_model.to(device)
detr_model.eval()
print("Modelo DETR cargado")

# Normalización para modelo de pose
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Tamaño que espera el modelo de pose
try:
    img_size = conf.MODEL.IMAGE_SIZE
    model_w, model_h = int(img_size[0]), int(img_size[1])
except Exception:
    model_w, model_h = 512, 384

# Skeleton EXACTO de la clase Visualizer (1-based)
SKELETON_1BASED = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], 
                   [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
                   [2, 4], [3, 5], [4, 6], [5, 7]]
SKELETON = [[a-1, b-1] for a, b in SKELETON_1BASED]  # Convertir a 0-based

# ---------- Funciones auxiliares iguales que app.py ----------
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
    """
    trans = get_affine_transform(center, scale, 0, output_size)
    img_np = np.array(img_pil)
    out = cv2.warpAffine(img_np, trans, (int(output_size[0]), int(output_size[1])), flags=cv2.INTER_LINEAR)
    return out

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

def get_pose_keypoints(person_crop_pil, bbox_expanded, pose_model, device, conf):
    """
    Estima los keypoints de una persona a partir de su recorte y los devuelve en
    coordenadas de la imagen original.

    PROCESO DE TRANSFORMACIÓN DE COORDENADAS:
    1. Imagen recortada → Transformación afín → Espacio del modelo (512x384)
    2. Predicción del modelo → Coordenadas en espacio del modelo
    3. Transformación inversa → Espacio del recorte original  
    4. Suma offset → Espacio de la imagen completa
    
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
    
    # Aplicar filtrado (usando mismo umbral que app.py)
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
    
    # Calcular plot_scale
    plot_scale = np.linalg.norm(scale) / 2.0
    
    return keypoints_original, filtered_scores.reshape(-1, 1), plot_scale

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
                print(f"NMS aplicado: {original_count} -> {filtered_count} personas (eliminadas {original_count - filtered_count} duplicados)")
        
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
                print(f"Error procesando keypoints para persona {person_count}: {e}")
        
        return image_with_keypoints, person_count, persons_details
        
    except Exception as e:
        print(f"Error en inferencia multi-persona: {e}")
        return None, 0, []

# ---------- Mapeo de índices COCO a nombres de keypoints ----------
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# ---------- Procesar todas las imágenes de input ----------
# input y output están en local/
input_dir = SCRIPT_DIR / "input"
output_dir = SCRIPT_DIR / "output"
output_dir.mkdir(exist_ok=True)

# Parámetros iguales que app.py
DETR_THRESHOLD = 0.9
NMS_THRESHOLD = 0.1
POINT_SIZE_MULTIPLIER = 1.0
LINE_WIDTH_MULTIPLIER = 1.0

# Obtener todas las imágenes del directorio input
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = []
for ext in image_extensions:
    image_files.extend(input_dir.glob(f"*{ext}"))
    image_files.extend(input_dir.glob(f"*{ext.upper()}"))

if len(image_files) == 0:
    print(f"No se encontraron imágenes en el directorio {input_dir}")
else:
    print(f"Procesando {len(image_files)} imagen(es) desde {input_dir}")
    print("=" * 60)

    for img_idx, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"Imagen {img_idx}/{len(image_files)}: {image_path.name}")
        print(f"{'='*60}")
        
        try:
            # Cargar imagen
            image = Image.open(image_path).convert("RGB")
            print(f"Tamaño de la imagen: {image.size}")
            
            # Realizar inferencia multi-persona
            print("Procesando personas y keypoints...")
            result_image, person_count, persons_details = infer_multi_person_pose(
                image, pose_model, detr_model, detr_processor, device, conf,
                POINT_SIZE_MULTIPLIER, LINE_WIDTH_MULTIPLIER, 
                DETR_THRESHOLD, NMS_THRESHOLD
            )
            
            if result_image is not None:
                # Guardar imagen con keypoints
                output_image_name = f"{image_path.stem}_pose.jpg"
                output_image_path = output_dir / output_image_name
                result_image.save(output_image_path)
                print(f"✅ Imagen guardada: {output_image_path}")
                
                # Guardar JSON con toda la información
                persons_data = []
                for p_idx, details in enumerate(persons_details, 1):
                    keypoints_json = []
                    for i, kp in enumerate(details['keypoints']):
                        kp_data = {
                            "name": COCO_KEYPOINTS[i] if i < len(COCO_KEYPOINTS) else f"keypoint_{i}",
                            "x": float(kp[0]) if not np.isnan(kp[0]) else None,
                            "y": float(kp[1]) if not np.isnan(kp[1]) else None,
                            "confidence": float(details['confidences'][i]) if i < len(details['confidences']) and not np.isnan(details['confidences'][i]) else 0.0,
                            "visible": not (np.isnan(kp[0]) or np.isnan(kp[1]))
                        }
                        keypoints_json.append(kp_data)
                    
                    person_data = {
                        "person_id": p_idx,
                        "detection": {
                            "bounding_box": {
                                "x_min": float(details['bbox'][0]),
                                "y_min": float(details['bbox'][1]),
                                "x_max": float(details['bbox'][2]),
                                "y_max": float(details['bbox'][3]),
                            }
                        },
                        "keypoints": keypoints_json
                    }
                    persons_data.append(person_data)
                
                json_output_name = f"{image_path.stem}_detections.json"
                json_output_path = output_dir / json_output_name
                json_data = {
                    "image_size": {
                        "width": int(image.size[0]),
                        "height": int(image.size[1])
                    },
                    "total_persons": len(persons_data),
                    "persons": persons_data
                }
                
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ JSON guardado: {json_output_path}")
                print(f"   Personas detectadas: {person_count}")
            else:
                print(f"❌ Error: No se pudo procesar la imagen {image_path.name}")
                
        except Exception as e:
            print(f"❌ Error procesando {image_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("✅ Procesamiento completado")
    print(f"{'='*60}")
