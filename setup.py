from setuptools import find_packages, setup
from typing import List

Hypen_e_dot = '-e .'
def get_reuirements(file_path:str)->List[str]:
    reuirements = []
    with open(file_path) as file:
        reuirements = file.readlines()
        reuirements = [req.replace("\n", "") for req in reuirements]
        if Hypen_e_dot in reuirements:
            reuirements.remove(Hypen_e_dot)
        return reuirements



setup(
    name = "DiamondPricePrediction",
    version="0.0.1",
    author="Shogun",
    author_email="shagunsharma029@gmail.com",
    install_requires = get_reuirements("requirements.txt"),
    packages=find_packages()


)