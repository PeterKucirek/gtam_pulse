from setuptools import setup, find_packages

setup(
    name="gtam_pulse",
    author="PeterKucirek",
    version="0.1-dev",
    packages=find_packages(),
    python_requires='>=3.6',

    requires=[
        'balsa',
        'numpy',
        'pandas',
        'attrs',
        'numexpr'
    ]
)


