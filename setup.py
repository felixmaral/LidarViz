from setuptools import setup, find_packages

setup(
    name="lidar_visualizer",
    version="0.3",
    packages=find_packages(where="src"),  
    package_dir={"": "src"},              
    install_requires=[
        "open3d",
        "numpy",
        "matplotlib",
        "plyfile",
        "pygame",
        "carla",
        "tk",  
    ],
    entry_points={
        "console_scripts": [
            "lidar-viz=main:main",  
        ],
    },
    include_package_data=True,  
    description="Proyecto de visualización de datos LiDAR con soporte para archivos y simulador CARLA",
    author="Félix Martínez",
    author_email="felixmaral131@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)