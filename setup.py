from setuptools import find_packages, setup


if __name__ == "__main__":
    setup(
        name="mlrap",
        version="1.0",
        description="Machine Learning Regression Analyse Packages",
        url="https://github.com/NianSan-H/mlrap",
        author="Gang Tang, Tao Hu, Chunbao Feng",
        license="MIT License",
        include_package_data=True,
        packages=find_packages(),
        package_data={
            "mlrap": [
                "config\run-CONFIG.yaml",
                "config\optim-CONFIG.yaml",
                "config\current.mplstyle",
                "config\shap.mplstyle",
                "example\dielectric_constant.json.gz",
                "example\elastic_tensor_2015.json.gz"
            ]
        },
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
        },
        python_requires=">=3.8",
    )