import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='magLabUtilities',
    version='1.0.1',
    author='Mark Travers',
    author_email='mark.2.travers@ucdenver.edu',
    description='Commonly used utilities in the UC Denver magnetics lab.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MarkTravers/magLabUtilities',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy==1.18.3'
        'astroid==2.4.2'
        'colorama==0.4.4'
        'dill==0.3.3'
        'isort==5.6.4'
        'lazy-object-proxy==1.4.3'
        'mccabe==0.6.1'
        'multiprocess==0.70.11.1'
        'pathos==0.2.7'
        'pox==0.2.9'
        'ppft==1.6.6.3'
        'pylint==2.6.0'
        'six==1.15.0'
        'toml==0.10.2'
        'typed-ast==1.4.1'
        'wrapt==1.12.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6'
)