import open3d as o3d
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys
from plyfile import PlyData

# ------ Parámetros y valores por defecto -------
colormaps_list = ['plasma', 'jet', 'inferno', 'viridis', 'cividis', 'turbo', 'coolwarm']
zoom_third_person = 0.012
zoom_top = 0.06

# ------- Funciones de Visualización -------

def read_bin_file(file_path):
    if not file_path.endswith('.bin') or not os.path.exists(file_path):
        print(f"Error: .bin file not found at {file_path}")
        sys.exit(1)
    print(file_path)
    scan = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
    points, remissions = scan[:, 0:3], scan[:, 3]
    return points, remissions

def read_ply_file(file_path):
    if not file_path.endswith('.ply') or not os.path.exists(file_path):
        print(f"Error: .ply file not found at {file_path}")
        sys.exit(1)
    plydata = PlyData.read(file_path)
    print(file_path)
    x, y, z = plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']
    points = np.vstack((x, y, z)).T
    remissions = plydata['vertex'].data['intensity']
    return points, remissions

def load_path_files(path) -> list:
    def extract_sample_number(file_name, extension):
        parts = file_name.split("__" if extension == '.bin' else "-")
        num_str = ''.join(filter(str.isdigit, parts[1] if extension == '.bin' else parts[0]))
        return int(num_str) if num_str.isdigit() else 0

    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.bin') or file.endswith('.ply')]
    if files:
        ext = os.path.splitext(files[0])[1]
        return sorted(files, key=lambda file: extract_sample_number(os.path.basename(file), ext))
    return []

def set_colors(remissions, colormap_name='plasma') -> np.ndarray:
    norm_remissions = (remissions - remissions.min()) / (remissions.max() - remissions.min())
    colors = plt.get_cmap(colormap_name)(norm_remissions)[:, :3]
    return colors

def update_colors(point_cloud, remissions, colormap_name='plasma'):
    colors = set_colors(remissions, colormap_name)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def add_new_sample(point_cloud, points, colors):
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def configure_render_options(vis):
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1.5

def configure_camera_third_person(vis, data_type=True):
    view_control = vis.get_view_control()
    view_control.set_front([-1, 0, 0.4] if data_type else [1, 0, 0.4])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(zoom_third_person)

def configure_camera_top(vis, data_type=True):
    view_control = vis.get_view_control()
    view_control.set_up([1, 0, 0] if data_type else [-1, 0, 0])
    view_control.set_front([0, 0, 1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_zoom(zoom_top)

def add_sensor(vis):
    # Añadir la posición del sensor como un cilindro rojo en el origen
    sensor_pos = o3d.geometry.TriangleMesh.create_cylinder(radius=0.2, height=0.5)  # Crear un cilindro
    sensor_pos.translate([0, 0, 0])  # Posición del sensor en el origen
    sensor_pos.paint_uniform_color([1, 0, 0])  # Color rojo para el sensor
    vis.add_geometry(sensor_pos)

def add_axis(vis):
    # Añadir ejes de coordenadas en el origen para referencia
    ejes_coordenadas = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])  # Tamaño ajustable
    vis.add_geometry(ejes_coordenadas)

def vis_sequences(path, initial_colormap='plasma', initial_fps=0.5):
    FPS = [initial_fps]
    current_colormap = [initial_colormap]
    path_file_list = load_path_files(path)
    is_bin_file = [path_file_list[0].endswith('.bin')] if path_file_list else [False]
    num_files = len(path_file_list)
    point_cloud = o3d.geometry.PointCloud()

    if is_bin_file[0]:
        points, remissions_data = read_bin_file(path_file_list[0])
    else:
        points, remissions_data = read_ply_file(path_file_list[0])

    remissions = [remissions_data]
    add_new_sample(point_cloud, points, set_colors(remissions[0], current_colormap[0]))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='PointCloud Sequence')
    configure_render_options(vis)
    vis.add_geometry(point_cloud)
    add_sensor(vis)
    add_axis(vis)
    configure_camera_third_person(vis, is_bin_file[0])
    

    is_third_person = [True]
    colormap_index = [colormaps_list.index(current_colormap[0])]
    background = [False]
    is_paused = [False]
    is_auto_mode = [True]
    is_resampled = [False]
    frame = [0]
    last_update_time = [time.time()]

    def toggle_camera(vis):
        if is_third_person[0]:
            configure_camera_top(vis, is_bin_file[0])
        else:
            configure_camera_third_person(vis, is_bin_file[0])
        is_third_person[0] = not is_third_person[0]
        vis.update_renderer()

    def toggle_colormap(vis):
        colormap_index[0] = (colormap_index[0] + 1) % len(colormaps_list)
        current_colormap[0] = colormaps_list[colormap_index[0]]
        update_colors(point_cloud, remissions[0], current_colormap[0])
        vis.update_geometry(point_cloud)
        vis.update_renderer()
        print(f'Colormap changed to: {current_colormap[0]}')

    def toggle_background(vis):
        vis.get_render_option().background_color = [0.05, 0.05, 0.05] if background[0] else [0.95, 0.95, 0.95]
        background[0] = not background[0]
        vis.update_renderer()

    def toggle_pause(vis):
        is_paused[0] = not is_paused[0]
        print("Paused" if is_paused[0] else "Playing")

    def increase_fps(vis):
        FPS[0] += 1
        print(f"FPS increased to: {FPS[0]}")

    def decrease_fps(vis):
        FPS[0] = max(0.1, FPS[0] - 1)
        print(f"FPS decreased to: {FPS[0]}")

    def toggle_mode(vis):
        is_auto_mode[0] = not is_auto_mode[0]
        print("Mode:", "Automatic" if is_auto_mode[0] else "Manual")

    def toggle_resampling(vis):
        is_resampled[0] = not is_resampled[0]
        print("Resampling:", "1:3" if is_resampled[0] else "Original")
        update_point_cloud()

    def next_frame():
        frame[0] = (frame[0] + 1) % num_files
        update_point_cloud()

    def prev_frame():
        frame[0] = (frame[0] - 1) % num_files
        update_point_cloud()

    def update_point_cloud():
        if is_bin_file[0]:
            points, remissions_data = read_bin_file(path_file_list[frame[0]])
        else:
            points, remissions_data = read_ply_file(path_file_list[frame[0]])
        if is_resampled[0]:
            points, remissions_data = points[::3], remissions_data[::3]
        remissions[0] = remissions_data
        add_new_sample(point_cloud, points, set_colors(remissions[0], current_colormap[0]))
        vis.update_geometry(point_cloud)
        vis.update_renderer()

    vis.register_key_callback(ord("V"), toggle_camera)
    vis.register_key_callback(ord("C"), toggle_colormap)
    vis.register_key_callback(ord("B"), toggle_background)
    vis.register_key_callback(32, toggle_pause)
    vis.register_key_callback(265, increase_fps)
    vis.register_key_callback(264, decrease_fps)
    vis.register_key_callback(ord("M"), toggle_mode)
    vis.register_key_callback(ord("N"), toggle_resampling)
    vis.register_key_callback(262, lambda vis: next_frame() if not is_auto_mode[0] else None)
    vis.register_key_callback(263, lambda vis: prev_frame() if not is_auto_mode[0] else None)

    def update_frame(vis):
        if is_paused[0] or not is_auto_mode[0]:
            return
        current_time = time.time()
        if current_time - last_update_time[0] >= 1 / FPS[0]:
            next_frame()
            last_update_time[0] = current_time

    vis.register_animation_callback(update_frame)
    vis.run()
    vis.destroy_window()