try:
    import carla
except ImportError:
    print("Módulo CARLA no encontrado. Asegúrate de que está instalado y configurado.")
    pass

import numpy as np
import open3d as o3d
import time
import random
from datetime import datetime
from matplotlib import colormaps as cm
import sys
import math
import pygame

import os
# Obtenemos la ruta absoluta de la carpeta que contiene este mismo archivo (carla_viz.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Añadimos esa carpeta a la lista de rutas donde Python busca módulos
if current_dir not in sys.path:
    sys.path.append(current_dir)

## INFERENCE: Importaciones necesarias para PyTorch y el modelo
import torch
from pointnet2_model import PointNet2SemSeg


# --- Configuración Visual y Global ---
VIRIDIS = np.array(cm.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

actor_list = []
manual_mode = False
absolute_view = False
camera_views = [
    {"zoom": 0.3, "front": [0, 0, 1], "lookat": [0, 0, 0], "up": [-1, 0, 0]},
    {"zoom": 0.06, "front": [1.0, 0.0, 0.3], "lookat": [0, 0, 0], "up": [0, 0, 1]},
    {"zoom": 0.2, "front": [0, 0, -1], "lookat": [0, 0, 0], "up": [0, 1, 0]}
]
current_view_index = 0

## INFERENCE: Variables globales para el modelo y la inferencia
inference_mode = False
inference_model = None
device = None
NUM_POINTS_FOR_INFERENCE = 8192 # Número de puntos que espera tu modelo
# Define los colores para cada clase. (Ej: 0=Fondo, 1=Vehículo, 2=Peatón)
# Puedes personalizarlos. Formato [R, G, B] normalizado.
# ESTA ES LA VERSIÓN CORRECTA CON 9 COLORES
CLASS_COLORS = np.array([
    [0.5, 0.5, 0.5],    # Clase 0 (Gris)
    [1.0, 0.0, 0.0],    # Clase 1 (Rojo)
    [0.0, 0.0, 1.0],    # Clase 2 (Azul)
    [0.0, 1.0, 0.0],    # Clase 3 (Verde)
    [1.0, 1.0, 0.0],    # Clase 4 (Amarillo)
    [1.0, 0.0, 1.0],    # Clase 5 (Magenta)
    [0.0, 1.0, 1.0],    # Clase 6 (Cian)
    [1.0, 0.5, 0.0],    # Clase 7 (Naranja)
    [0.5, 0.0, 1.0]     # Clase 8 (Púrpura)
])

# --- Funciones de Inferencia ---

## INFERENCE: Carga el modelo PointNet++
def load_inference_model(model_path, num_classes=9):
    global inference_model, device
    print("Cargando modelo de inferencia...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {device}")
        
        inference_model = PointNet2SemSeg(num_classes=num_classes)
        inference_model.load_state_dict(torch.load(model_path, map_location=device))
        inference_model.to(device)
        inference_model.eval() # Poner el modelo en modo de evaluación
        print("Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        inference_model = None

## INFERENCE: Ejecuta la inferencia en una nube de puntos
def run_inference(points_np):
    if inference_model is None:
        return None, None

    # 1. Preprocesar los puntos: Muestrear/ajustar al tamaño de entrada del modelo
    num_points = points_np.shape[0]
    if num_points > NUM_POINTS_FOR_INFERENCE:
        # Muestreo aleatorio si hay más puntos de los necesarios
        indices = np.random.choice(num_points, NUM_POINTS_FOR_INFERENCE, replace=False)
    else:
        # Duplicar puntos si hay menos de los necesarios
        indices = np.random.choice(num_points, NUM_POINTS_FOR_INFERENCE, replace=True)
    
    sampled_points = points_np[indices, :]

    # 2. Convertir a Tensor de PyTorch y ajustar dimensiones
    points_tensor = torch.from_numpy(sampled_points).float().to(device)
    points_tensor = points_tensor.unsqueeze(0)  # Añadir dimensión de batch -> (1, N, 3)
    points_tensor = points_tensor.transpose(2, 1) # Cambiar a (1, 3, N)

    # 3. Realizar la inferencia
    with torch.no_grad():
        output = inference_model(points_tensor)
    
    # 4. Obtener las etiquetas predichas
    pred_labels = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    return sampled_points, pred_labels

# --- Funciones CARLA y de Visualización ---

def euler_to_rotation_matrix(pitch, yaw, roll):
    pitch, yaw, roll = map(math.radians, [pitch, yaw, roll])
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return R_z @ R_y @ R_x

## INFERENCE: Modificado para gestionar el modo de inferencia
def lidar_callback(lidar_data, point_cloud, vehicle_transform):
    data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    data[:, 0] = -data[:, 0]

    if absolute_view:
        vehicle_location = np.array([-vehicle_transform.location.x, vehicle_transform.location.y, vehicle_transform.location.z])
        rotation_matrix = euler_to_rotation_matrix(
            vehicle_transform.rotation.pitch, 
            vehicle_transform.rotation.yaw, 
            vehicle_transform.rotation.roll
        ).T
        points_in_world = (rotation_matrix @ data[:, :3].T).T + vehicle_location
        points_to_display = points_in_world
    else:
        points_to_display = data[:, :3]

    if inference_mode:
        ## INFERENCE: Si está activo, ejecutar inferencia y colorear por clase
        sampled_points, pred_labels = run_inference(points_to_display)
        
        if pred_labels is not None:
            # Mapear etiquetas a colores
            seg_colors = CLASS_COLORS[pred_labels]
            # Actualizar tanto los puntos (muestreados) como los colores
            point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
            point_cloud.colors = o3d.utility.Vector3dVector(seg_colors)
        else:
            # Si hay un error, mostrar en negro
            point_cloud.points = o3d.utility.Vector3dVector(points_to_display)
            point_cloud.colors = o3d.utility.Vector3dVector(np.zeros_like(points_to_display))
    else:
        ## INFERENCE: Comportamiento original, colorear por intensidad
        intensity = data[:, -1]
        int_color = np.c_[
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 2])
        ]
        point_cloud.points = o3d.utility.Vector3dVector(points_to_display)
        point_cloud.colors = o3d.utility.Vector3dVector(int_color)

def spawn_vehicle_lidar_camera(world, bp, traffic_manager, delta, lidar_range=100, channels=64, points_per_second=1200000):
    vehicle_bp = bp.filter('vehicle.*')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    lidar_bp = bp.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', str(lidar_range))
    lidar_bp.set_attribute('rotation_frequency', str(1 / delta))
    lidar_bp.set_attribute('channels', str(channels))
    lidar_bp.set_attribute('points_per_second', str(points_per_second))
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=-0.5, z=1.8)), attach_to=vehicle)

    camera_bp = bp.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=-4.0, z=2.5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    
    vehicle.set_autopilot(True, traffic_manager.get_port())
    return vehicle, lidar, camera

def create_origin_sphere():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    return sphere

def set_camera_view(viz, view_index):
    ctr = viz.get_view_control()
    view = camera_views[view_index]
    ctr.set_zoom(view["zoom"])
    ctr.set_front(view["front"])
    ctr.set_lookat(view["lookat"])
    ctr.set_up(view["up"])

def camera_callback(image, display_surface):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    array = array[:, :, :3][:, :, ::-1] # BGRA a RGB
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display_surface.blit(surface, (0, 0))
    pygame.display.update(display_surface.get_rect())

def vehicle_control(vehicle):
    global manual_mode
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            manual_mode = not manual_mode
            vehicle.set_autopilot(not manual_mode)
            print(f"\nCoche en modo {'manual' if manual_mode else 'automático'}.")
    
    if manual_mode:
        keys = pygame.key.get_pressed()
        control = carla.VehicleControl(
            throttle=1.0 if keys[pygame.K_w] else 0.0,
            brake=1.0 if keys[pygame.K_s] else 0.0,
            steer=-0.3 if keys[pygame.K_a] else 0.3 if keys[pygame.K_d] else 0.0
        )
        vehicle.apply_control(control)

def main(lidar_range, channels, points_per_second):
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.SRCALPHA)
    pygame.display.set_caption("CARLA Vehículo Control")

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    ## INFERENCE: Cargar el modelo al inicio
    # Asegúrate de que esta ruta sea correcta dentro de tu contenedor Docker si es necesario
    model_path = "/home/carla/2024-tfg-felix-martinez/segmentation/deep_learning/results/pointnet++_no_rem_v0/pointnet2_no_remission_v1.pth"
    load_inference_model(model_path)

    original_settings = world.get_settings()
    
    try:
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        
        settings = world.get_settings()
        delta = 0.05
        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        world.apply_settings(settings)

        global actor_list
        vehicle, lidar, camera = spawn_vehicle_lidar_camera(world, world.get_blueprint_library(), traffic_manager, delta, lidar_range, channels, points_per_second)
        actor_list.extend([vehicle, lidar, camera])

        camera.listen(lambda image: camera_callback(image, screen))
        point_cloud = o3d.geometry.PointCloud()
        lidar.listen(lambda data: lidar_callback(data, point_cloud, lidar.get_transform()))

        viz = o3d.visualization.VisualizerWithKeyCallback()
        viz.create_window(window_name='Lidar simulado en Carla', width=960, height=540, left=480, top=270)
        viz.get_render_option().background_color = [0.05, 0.05, 0.05]
        viz.get_render_option().point_size = 1.35

        set_camera_view(viz, current_view_index)
        origin_sphere = create_origin_sphere()
        origin_sphere_added = False

        def toggle_camera_view(vis):
            nonlocal origin_sphere_added
            global current_view_index, absolute_view
            current_view_index = (current_view_index + 1) % len(camera_views)
            set_camera_view(vis, current_view_index)
            absolute_view = (current_view_index == len(camera_views) - 1)
            if absolute_view and not origin_sphere_added:
                vis.add_geometry(origin_sphere)
                origin_sphere_added = True
            elif not absolute_view and origin_sphere_added:
                vis.remove_geometry(origin_sphere)
                origin_sphere_added = False
            print(f"Cambiando a vista {current_view_index + 1}")

        ## INFERENCE: Callback para activar/desactivar la inferencia
        def toggle_inference_mode(vis):
            global inference_mode
            if inference_model is None:
                print("\nNo se puede activar el modo inferencia: el modelo no se ha cargado.")
                return
            inference_mode = not inference_mode
            mode_str = "ACTIVADO" if inference_mode else "DESACTIVADO"
            print(f"\nModo Inferencia: {mode_str}")

        viz.register_key_callback(ord("V"), toggle_camera_view)
        viz.register_key_callback(ord("I"), toggle_inference_mode) ## INFERENCE: Registrar la tecla 'I'

        frame, lidar_added = 0, False
        while True:
            world.tick()
            vehicle_control(vehicle)
            
            if not lidar_added and len(point_cloud.points) > 0:
                viz.add_geometry(point_cloud)
                lidar_added = True

            viz.update_geometry(point_cloud)
            if not viz.poll_events():
                break
            viz.update_renderer()

    except SystemExit:
        print("Cerrando pygame y saliendo.")
    
    finally:
        print("\nLimpiando...")
        world.apply_settings(original_settings)
        for actor in actor_list:
            if actor and actor.is_alive:
                actor.destroy()
        actor_list.clear()
        pygame.quit()
        print("Limpieza completa.")