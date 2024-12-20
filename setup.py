from setuptools import find_packages, setup

setup(
    name='MLP_CW3',
    packages=find_packages(),
    version='0.0.0',
    install_requires=[
        'click==8.0.3',
        'cpprb==10.6.3',
        'einops==0.7.0',
        'hydra-core==1.0.5',
        'gym==0.21.0',
        'h5py==3.10.0',
        'info-nce-pytorch==0.1.4',
        'matplotlib',
        'omegaconf==2.0.6',
        'pyvirtualdisplay',
        'pyglet==1.5.21',
        'pyyaml==5.4.1',
        "seaborn==0.13.0",
        "setuptools==62.1.0",
        'stable-baselines3==1.8.0',
        'tensorboard',
        'torch==2.1.0',
        'tqdm',
        'imageio==2.13.1',
        'imageio-ffmpeg==0.4.5',
        'wandb==0.16.3',
        'torch-geometric==2.5.0'
    ]
)