from setuptools import setup, find_packages

setup(
    name='my_krml_25176165',
    version='2025.0.3.1',
    author='Nian-Ya,Weng',
    author_email='Nian-Ya.Weng@student.uts.edu.au',
    description='My experiment package for TestPyPI',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/Amanda1005/my_krml_25176165',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)



