import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='magLabUtilities',
    version='1.0.0',
    author='Mark Travers',
    author_email='mark.2.travers@ucdenver.edu',
    description='Commonly used utilities in the UC Denver magnetics lab.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MarkTravers/magLabUtilities',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.18.3'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6'
)