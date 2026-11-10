from glob import glob

from setuptools import find_packages, setup

package_name = "adp_control"
setup(
    name=package_name, version="1.0.0", packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"], zip_safe=True,
    entry_points={"console_scripts": ["controller_node = adp_control.controller_node:main", "metrics_node = adp_control.metrics_node:main"]},
)
