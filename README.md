## Run it 

### Terminal 1:
```bash . install/setup.bash
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia 
```
```bash export GZ_SIM_RESOURCE_PATH="$HOME/ardu_ws/install/ardupilot_gazebo/share:$GZ_SIM_RESOURCE_PATH"
export SDF_PATH="$GZ_SIM_RESOURCE_PATH"
```
```bash ros2 launch ardupilot_gz_bringup iris_maze.launch.py rviz:=true use_gz_tf:=true
```

### Terminal 2:
```bash . install/setup.bash
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true slam_params_file:=/home/$USER/ardu_ws/config/mapper_params_online_async.yaml
```

### Terminal 3:
```bash . install/setup.bash
 ros2 run frontier_exploration frontier_detector
```

### Terminal 4:
```bash . install/setup.bash
ros2 run frontier_exploration frontier_selector 
```

### Terminal 5:
```bash . install/setup.bash
ros2 run frontier_exploration astar_planner
```

### Terminal 6:
```bash  . install/setup.bash
ros2 run frontier_exploration frontier_controller
```

### Terminal 7:
```bash . install/setup.bash
mavproxy.py --master udp:127.0.0.1:14550  --console
```
