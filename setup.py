from setuptools import setup, find_packages

setup(
    name="aero-simulation-env",
    version="0.1.0",
    description="Quanser Aero 2 Simulation Environment for Reinforcement Learning",
    author="Georg Schäfer",
    author_email="georg.schaefer@fh-salzburg.ac.at",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gymnasium>=0.26.0",
        "numpy>=1.21.0",
        "pygame>=2.0.0",
    ],
    python_requires='>=3.8',
)
