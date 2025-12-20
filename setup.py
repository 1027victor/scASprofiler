from setuptools import setup, find_packages

setup(
    name="scASprofiler",
    version="0.1.0",
    description="A Deep Convolutional Generative Network for Single-cell Alternative Splicing Profiler",
    packages=find_packages(),  # 会自动找到 scASprofiler 及其子包
    include_package_data=True,
    python_requires=">=3.10.9",
    # install_requires=[
    #     "click>=8.0",
    #     # 按你实际用到的填：
    #     "numpy",
    #     "pandas",
    #     "scipy",
    #     "scikit-learn",
    #     "torch",
    #     # ...
    # ],
    entry_points={
        "console_scripts": [
            # 3 个独立命令
            "scASprofiler-impute=scASprofiler.scASp_impute.cli:cli",
            "scASprofiler-perp=scASprofiler.scASp_perp.cli:cli",
            "scASprofiler-quantify=scASprofiler.scASp_quantify.calculate_as_ratio:main",
        ]
    },
)
