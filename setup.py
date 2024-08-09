from setuptools import find_packages, setup

setup(
    name="scikit-jax",
    version="0.0.1",
    author="Liiban Mohamud",
    author_email="liibanmohamud12@gmail.com",
    description="Classical machine learning algorithms on the GPU.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LiibanMo/scikit-jax",
    packages=find_packages(),
    install_requires=[
        "jax",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="jax classical machine learning",
    python_requires=">=3.9",
)
