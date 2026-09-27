from setuptools import find_packages, setup
import os
import glob

package_name = 'llm_coordinator'

# ---------------------------------------------------------------
# Collect data files: knowledge_base/*.txt  and  config/*.yaml
# These get installed to share/llm_coordinator/knowledge_base/
# and share/llm_coordinator/config/ so the node can find them.
# ---------------------------------------------------------------
knowledge_files = glob.glob('knowledge_base/*.txt')
config_files = glob.glob('config/*.yaml')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Knowledge base files
        ('share/' + package_name + '/knowledge_base', knowledge_files),

        # Robot config file
        ('share/' + package_name + '/config', config_files),
    ],
    install_requires=[
        'setuptools',
        'pyyaml',
        'requests',
    ],
    zip_safe=True,
    maintainer='Sentoor Kumar',
    maintainer_email='sentoorkumar@gmail.com',
    description='LLM Coordinator with Knowledge-Grounded Decision Making',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'llm_node = llm_coordinator.llm_node:main',
        ],
    },
)