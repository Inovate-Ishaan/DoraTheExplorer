from setuptools import find_packages, setup

package_name = 'frontier_exploration'

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
    maintainer='krish',
    maintainer_email='ishaanshukla.rn@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'frontier_detector = frontier_exploration.frontier_detector:main',
            'frontier_selector = frontier_exploration.frontier_selector:main',
            'astar_planner = frontier_exploration.astar_planner:main',
            'frontier_controller = frontier_exploration.frontier_controller:main',
        ],
    },
)
