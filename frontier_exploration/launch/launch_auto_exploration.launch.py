from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    slam_toolbox_launch_file = os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')
    slam_params = os.path.join(os.path.expanduser('~'), 'ardu_ws', 'config',
                               'mapper_params_online_async.yaml')

    include_slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_toolbox_launch_file),
        launch_arguments={
            'use_sim_time' : 'true',
            'slam_params_file' : slam_params
        }.items()
    )

    return LaunchDescription([
        include_slam_launch,
        
        Node(
            package='frontier_exploration',
            executable='frontier_detector',
            name='frontier_detector',
            parameters= [{'use_sim_time' : True}]
        ),
        Node(
            package='frontier_exploration',
            executable='frontier_selector',
            name='frontier_selector',
            parameters= [{'use_sim_time' : True}]
        ),
        Node(
            package='frontier_exploration',
            executable='astar_planner',
            name='astar_planner',
            parameters= [{'use_sim_time' : True}]

        ),
        Node(
            package='frontier_exploration',
            executable='mission_manager',
            name='mission_manager',
            parameters= [{'use_sim_time' : True}]
        ),
        Node(
            package='frontier_exploration',
            executable='frontier_controller',
            name='frontier_controller',
            parameters= [{'use_sim_time' : True}]
        ),
        Node(
            package='frontier_exploration',
            executable='precision_lander',
            name='precision_lander',
            parameters= [{'use_sim_time' : True}]
        )
    ])