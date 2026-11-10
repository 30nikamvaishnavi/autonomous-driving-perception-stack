from glob import glob

from setuptools import find_packages, setup

package_name = "adp_perception"
setup(
    name=package_name, version="1.0.0", packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"], zip_safe=True,
    entry_points={"console_scripts": ["detector_node = adp_perception.detector_node:main", "tracker_node = adp_perception.tracker_node:main", "lane_node = adp_perception.lane_node:main", "depth_node = adp_perception.depth_node:main", "fusion_node = adp_perception.fusion_node:main"]},
)
