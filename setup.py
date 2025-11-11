from setuptools import setup, find_packages

setup(
    name="deepsdf",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "deepsdf": ["py.typed"],
    },
)
