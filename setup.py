from setuptools import find_packages, setup

package_name = "univla_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@todo.todo",
    description="ROS2 wrapper for UniVLA VLA model inference",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            f"univla_node = {package_name}.univla_node:main",
        ],
    },
)
