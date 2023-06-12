from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='dep_tree_parser',
      version='0.0.1',
      url='http://github.com/fractalego/tree_parser',
      author='Alberto Cetoli',
      author_email='alberto@nlulite.com',
      description="A programmable relation extractor",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['tree_parser'],
      install_requires=[
          'numpy==1.18.1',
          'transformers==4.30.0',
          'python-igraph==0.7.1.post6',
          'pytorch-transformers==1.2.0',
          'networkx==2.4.0',
          'scipy==1.2.3',
      ],
      classifiers=[
          'License :: OSI Approved :: MIT License',
      ],
      include_package_data=True,
      zip_safe=False)
