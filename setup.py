from setuptools import setup, find_packages

setup(
    name="scASprofiler",
    version="0.1.0",
    description="A Deep Convolutional Generative Network for Single-cell Alternative Splicing Profiler",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10.9",
    entry_points={
        "console_scripts": [
            "scASprofiler-impute=scASprofiler.scASp_impute.cli:cli",
            "scASprofiler-perp=scASprofiler.scASp_perp.cli:cli",
            "scASprofiler-quantify=scASprofiler.scASp_quantify.calculate_as_ratio:main",
        ]
    },
)
