from setuptools import find_packages, setup

package_name = 'speech_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
    'setuptools',
    'sounddevice',
    'scipy',
    'openai-whisper',
    'torch',],
    zip_safe=True,
    maintainer='Sentoor Kumar',
    maintainer_email='sentoorkumar@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stt_node = speech_interface.stt_node:main',
            'tts_node = speech_interface.tts_node:main',
        ],
    },
)
