from setuptools import setup, find_packages

setup(
    name="your-backend-name",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "langchain",
        "langchain-core",
        # ... other dependencies ...
    ],
)