from setuptools import setup
from setuptools import find_packages


setup(
    name="speelysis",
    version="0.1.0",
    license="MIT",
    description="音声分析",
    author="conf8o",
    url="https://github.com/conf8o",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib"
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"]
)
