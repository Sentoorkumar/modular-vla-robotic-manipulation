from setuptools import setup
import os
from glob import glob

package_name = 'perception_agent'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools',
        'ultralytics',
        'numpy',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='Sentoor Kumar',
    maintainer_email='sentoorkumar@gmail.com',
    description='Perception agent for object detection using YOLO',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = perception_agent.perception_node:main',
        ],
    },
)
