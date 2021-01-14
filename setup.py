import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='magLabUtilities',
    version='1.0.7',
    author='Mark Travers',
    author_email='mark.2.travers@ucdenver.edu',
    description='Commonly used utilities in the UC Denver magnetics lab.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MarkTravers/magLabUtilities',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'dill==0.3.3',
        'multiprocess==0.70.11.1',
        'numpy==1.19.5',
        'pandas==1.2.0',
        'pathos==0.2.7',
        'pox==0.2.9',
        'ppft==1.6.6.3',
        'python-dateutil==2.8.1',
        'pytz==2020.5',
        'six==1.15.0',
        'openpyxl==3.0.5'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6'
)