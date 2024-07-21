from setuptools import setup, find_packages

setup(
    name="dlspark",
    version="1.0.4",
    description="A simple deep learning framework with NumPy backend.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Eric Lin",
    author_email="erix025@outlook.com",
    url="https://github.com/Erix025/dlspark",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
    install_requires=[
        'numpy>=2.0.0',
        'matplotlib>=3.4.3',
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.4",
        ],
    },
)