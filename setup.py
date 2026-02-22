"""
SAPPHIRE Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
with open(readme_file, encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
with open(requirements_file) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="sapphire-sc",
    version="1.0.0",
    author="SAPPHIRE Development Team",
    author_email="yexxx399@umn.edu@umn.edu",
    description="Single-cell Analysis of Pathway Plasticity via Heterogeneity-Informed Regulatory Entropy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ziye003/SAPPHIRE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
