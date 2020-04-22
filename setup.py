from setuptools import setup, find_packages

setup(
    name="gtam_pulse",
    author="PeterKucirek",
    version="0.1.0",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'balsa',
        'numpy',
        'pandas',
        'attrs',
        'numexpr'
    ]
)
