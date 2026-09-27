from setuptools import find_packages, setup

package_name = 'motion_agent'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sentoor Kumar',
    maintainer_email='sentoorkumar@gmail.com',
    description='Motion agent: MoveIt2 to Isaac Sim trajectory bridge and grasp coordinator',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'trajectory_bridge = motion_agent.trajectory_bridge:main',
            'motion_agent_node = motion_agent.motion_agent_node:main',
        ],
    },
)
