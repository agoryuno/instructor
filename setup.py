from setuptools import setup

setup(
    name='instructor',
    version='0.2.0.13',
    packages=['instructor'],
    url='',
    license='',
    author='',
    description='',
    requires_python='>=3.6',
    install_requires=[
        "datasets==2.8.0",
        "evaluate==0.4.0",
        "scikit-learn==1.2.0",
        "torch",
        "transformers==4.25.1",
        "wandb==0.13.7",
        "protobuf==3.19.*"
        ]
    )
