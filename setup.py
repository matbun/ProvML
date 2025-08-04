from setuptools import setup, find_packages
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Tmp fix for: https://github.com/wookayin/gpustat/issues/178
os.environ["GPUSTAT_VERSION"] = "1.1.1"

setup(
    name='prov4ml',
    version='1.0.0',
    packages=find_packages(),
    install_requires=required,  # Loaded from requirements.txt
    extras_require={
        'apple': [
            # Optional dependencies for Apple/Mac
            'apple_gpu==0.3.*'
        ], 
        'amd': [
            # Optional dependencies for AMD
            'pyamdgpuinfo==2.1.*',
        ], 
        'nvidia': [
            # Optional dependencies for NVIDIA
            'nvitop==1.3.*',
            'gpustat==1.1.1', # see os.environ["GPUSTAT_VERSION"] above
        ]
    }
)
