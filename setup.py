"""
Setup script for ReDimNet-MRL package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="redimnet-mrl",
    version="0.1.0",
    author="ReDimNet-MRL Contributors",
    description="Matryoshka Representation Learning for ReDimNet Speaker Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/redimnet-mrl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.20.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
        "tensorboard>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "wandb": [
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mrl-train=train:main",
        ],
    },
)
