from setuptools import setup, find_packages

setup(
    name="tcm_tongue",
    version="0.1.0",
    description="TCM Tongue Diagnosis Detection System",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[],
)
