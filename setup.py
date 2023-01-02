from setuptools import setup

setup(
    name='instructor',
    version='0.1',
    packages=['instructor'],
    url='',
    license='',
    author='',
    description='',
    requires_python='>=3.6',
    install_requires=[
        'datasets',
        'evaluate',
        'scikit-learn',
        'torch',
        'transformers',
        'wandb'
    ]
)
