import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
from lidar_visualization.file_viz import vis_sequences
from lidar_visualization.carla_viz import main as carla_main

# ------ global variables -------
colormaps_list = ['plasma', 'jet', 'inferno', 'viridis', 'cividis', 'turbo', 'coolwarm']
zoom_third_person = 0.012
zoom_top = 0.06

# ------- Interfaz gráfica para definir los parámetros iniciales -------
def initial_choice():
    # Configuración de la GUI inicial con dos botones
    root = tk.Tk()
    root.title("Seleccionar fuente de datos")
    root.geometry("300x200")
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use("clam")  # Tema moderno para ttk

    def launch_interface():
        def show_controls_window():
            # Crear una nueva ventana con Tkinter en un hilo separado
            def controls_window():
                controls = tk.Tk()
                controls.title("Controles de Visualización")
                controls.geometry("400x400")
                
                # Instrucciones de control en etiquetas
                controls_list = [
                    ("V", "Cambiar entre vista en tercera persona y vista superior"),
                    ("C", "Cambiar el colormap"),
                    ("B", "Cambiar color de fondo"),
                    ("M", "Alternar entre modo automático y manual"),
                    ("N", "Alternar entre muestreo 1:3 y original"),
                    ("Derecha", "Ir al siguiente fotograma (modo manual)"),
                    ("Izquierda", "Ir al fotograma anterior (modo manual)"),
                    ("Arriba", "Aumentar FPS"),
                    ("Abajo", "Disminuir FPS"),
                    ("Espacio", "Pausar/Reanudar (modo automático)"),
                ]
                
                tk.Label(controls, text="Controles:", font=("Arial", 14, "bold")).pack(pady=10)
                for key, description in controls_list:
                    tk.Label(controls, text=f"{key}: {description}", font=("Arial", 10)).pack(anchor="w", padx=20)
                    
                controls.mainloop()

            # Ejecuta la ventana de controles en un hilo separado
            threading.Thread(target=controls_window).start()

        def start_visualization():
            # Obtener valores de la GUI
            path = path_entry.get()
            colormap = colormap_var.get()
            fps = float(fps_var.get())
            
            # Validar el directorio
            if not os.path.isdir(path):
                messagebox.showerror("Error", "El directorio seleccionado no es válido.")
                return
            
            # Mostrar controles
            show_controls_window()
            root.destroy()
            
            # Lanzar la visualización
            vis_sequences(path, colormap, fps)

        # Configuración de la GUI con Tkinter
        root = tk.Tk()
        root.title("Configuración del Visor LiDAR")
        root.geometry("520x400")
        root.resizable(False, False)

        # Estilos personalizados
        style = ttk.Style()
        style.theme_use("clam")  # Tema moderno para ttk
        style.configure("TLabel", font=("Arial", 10), padding=5)
        style.configure("TButton", font=("Arial", 10, "bold"), padding=5)
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=5)

        # Contenedor principal
        frame = ttk.Frame(root, padding="20")
        frame.pack(fill="both", expand=True)

        # Campo de selección de directorio
        ttk.Label(frame, text="Selecciona el Directorio de Datos:").grid(row=0, column=0, sticky="w")
        path_entry = ttk.Entry(frame, width=40)
        path_entry.grid(row=1, column=0, padx=(0, 10), pady=5)
        ttk.Button(frame, text="Examinar", command=lambda: path_entry.insert(0, filedialog.askdirectory())).grid(row=1, column=1, pady=5)

        # Selección de Colormap
        ttk.Label(frame, text="Selecciona el Colormap:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        colormap_var = tk.StringVar(value=colormaps_list[0])
        colormap_dropdown = ttk.Combobox(frame, textvariable=colormap_var, values=colormaps_list, state="readonly")
        colormap_dropdown.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")

        # Selección de FPS con Slider
        ttk.Label(frame, text="FPS iniciales:").grid(row=4, column=0, sticky="w", pady=(10, 0))

        # Variable para el valor del slider y etiqueta para mostrar el valor actual
        fps_var = tk.IntVar(value=1)  # Inicializamos en 1
        fps_slider = ttk.Scale(frame, from_=1, to=20, orient="horizontal", variable=fps_var)  # Slider de 1 a 20
        fps_slider.grid(row=5, column=0, columnspan=1, pady=5, sticky="ew")

        # Etiqueta que muestra el valor seleccionado
        fps_value_label = ttk.Label(frame, text=f"{fps_var.get()} FPS")  # Inicializa con el valor actual
        fps_value_label.grid(row=5, column=1, padx=10, sticky="w")

        # Función para actualizar la etiqueta con el valor del slider
        def update_fps_label(*args):
            fps_value_label.config(text=f"{fps_var.get()} FPS")

        # Vinculamos el cambio de valor en el slider a la actualización de la etiqueta
        fps_var.trace("w", update_fps_label)

        # Botón para iniciar
        start_button = ttk.Button(frame, text="Iniciar Visualización", command=start_visualization)
        start_button.grid(row=6, column=0, columnspan=2, pady=(20, 0))

        root.mainloop()

    def on_carla_selected():
        root.destroy()  # Cierra la ventana inicial
        # Crear la ventana de configuración de parámetros del sensor
        config_window = tk.Tk()
        config_window.title("Configuración de Parámetros del Sensor")
        config_window.geometry("400x400")

        # Variables para los parámetros
        range_var = tk.StringVar(value="100")
        channels_var = tk.StringVar(value="64")
        pps_var = tk.StringVar(value="1200000")  # Puntos por segundo

        # Campos de entrada para los parámetros
        ttk.Label(config_window, text="Rango del LiDAR (metros):").pack(pady=5)
        range_entry = ttk.Entry(config_window, textvariable=range_var)
        range_entry.pack(pady=5)

        ttk.Label(config_window, text="Canales del LiDAR:").pack(pady=5)
        channels_entry = ttk.Entry(config_window, textvariable=channels_var)
        channels_entry.pack(pady=5)

        ttk.Label(config_window, text="Puntos por segundo (PPS):").pack(pady=5)
        pps_entry = ttk.Entry(config_window, textvariable=pps_var)
        pps_entry.pack(pady=5)

        def start_carla_with_params():
            # Obtener los valores actualizados de las entradas
            lidar_range = range_var.get()
            channels = channels_var.get()
            points_per_second = pps_var.get()

            # Imprimir para depurar y verificar los valores ingresados
            print(f"Starting CARLA with parameters: "
                f"Range={lidar_range}, Channels={channels}, PPS={points_per_second}")

            try:
                # Llamar a carla_main() con los parámetros convertidos a los tipos adecuados
                config_window.destroy()  # Cierra la ventana de configuración si todo va bien
                carla_main(
                    float(lidar_range), 
                    int(channels), 
                    int(points_per_second)
                )
            except ValueError as e:
                messagebox.showerror("Error de entrada", f"Por favor, ingrese valores válidos.\nError: {e}")

        # Botón de iniciar visualización
        start_button = ttk.Button(config_window, text="Iniciar Visualización", command=start_carla_with_params)
        start_button.pack(pady=20)

        config_window.mainloop()

    def on_files_selected():
        root.destroy()  # Cierra la ventana inicial
        launch_interface()  # Lanza la interfaz completa para archivos

    # Botones de selección
    ttk.Button(root, text="Carla Simulator", command=on_carla_selected).pack(expand=True, pady=5)
    ttk.Button(root, text="Files", command=on_files_selected).pack(expand=True, pady=5)

    root.mainloop()