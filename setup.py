from setuptools import find_packages, setup


if __name__ == "__main__":
    setup(
        name="mlrap",
        version="1.0",
        include_package_data=True,
        packages=find_packages(),
        install_requires=[
            "click",
            "scikit-learn",
            "xgboost",
            "pymatgen",
            "seaborn",
            "bayesian-optimization",
            "matminer",
            "shap",
            "CBFV",
            "xenonpy",
            "rdkit",
            "mlxtend",
        ],
        entry_points={
            "console_scripts": [
                "mlrap = mlrap.command.cli:cli",
            ]
        }
    )