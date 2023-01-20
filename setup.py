from setuptools import setup

setup(
    name='hl-run-manager',
    description='Manage, schedule and run Hamiltonian learning processes.',
    author='Frederik Wilde',
    author_email='wilde.pysics@gmail.com',
    url=None,
    license=None,
    python_requires='>=3.7',
    packages=['run_manager'],
    install_requires=[
        'jax >=0.2.9',
        'scipy',
        'sqlalchemy'
    ],
)