from setuptools import setup, find_packages

setup(
    name="cpas",
    version="1.2.0",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.0.0",
        "textblob>=0.15.3",
        "matplotlib>=3.4.0",
        "numpy>=1.20.0",
        "networkx>=2.5.0",
        "pyyaml>=5.4.0"
    ],
    extras_require={
        "full": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "scikit-learn>=0.24.0",
            "pandas>=1.3.0"
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0"
        ]
    },
    python_requires=">=3.8",
    author="CPAS Development Team",
    author_email="support@cpas-system.com",
    description="Comprehensive Psychological Analysis System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourrepo/cpas-core",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10"
    ]
) 