from setuptools import find_namespace_packages, setup


setup(
    name="instant-property-valuation",
    version=0.1,
    description="Estimate housing price in Florianópolis.",
    author="Eric Yang",
    author_email="ericyang.seed@gmail.com",
    python_requires=">=3.8",
    packages=find_namespace_packages(),
    #install_requires=[required_packages],
)

# Run this to install the package locally
#pip install -e .

# get the requirements
# pip freeze > requirements.txt
