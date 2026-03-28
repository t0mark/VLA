from setuptools import find_packages, setup

package_name = "na_vila_ros"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/navila.yaml"]),
        ("share/" + package_name + "/launch", ["launch/navila.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="todo@todo.todo",
    description="NaVILA VLA inference node for ROS2",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "navila_node = na_vila_ros.navila_node:main",
            "chat_prompt_node = na_vila_ros.chat_prompt_node:main",
            "regex_parshing_node = na_vila_ros.regex_parshing_node:main",
        ],
    },
)
