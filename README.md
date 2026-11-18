## Run it 

### Important Note -
Leftovers may cause unexpexted issues like DDS failures, sim_time mismatch, etc. First run this to clear all stale processes.
```bash
pkill -9 -f micro_ros_agent; pkill -9 -f MicroXRCEAgent; pkill -9 -f arducopter
pkill -9 -f mavproxy; pkill -9 -f sim_vehicle; pkill -9 -f "gz sim"; pkill -9 -f ruby; pkill -9 -f rviz2
ros2 daemon stop && ros2 daemon start
```


### Terminal 1:
```bash
 . install/setup.bash
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia 
```
```bash
export GZ_SIM_RESOURCE_PATH="$HOME/ardu_ws/install/ardupilot_gazebo/share:$HOME/ardu_ws/src/ardupilot_gz/ardupilot_gz_description/models:$GZ_SIM_RESOURCE_PATH"
export SDF_PATH="$GZ_SIM_RESOURCE_PATH"
```
```bash
. install/setup.bash
ros2 launch ardupilot_gz_bringup iris_maze.launch.py rviz:=true use_gz_tf:=true
```

### Terminal 2:
```bash
. install/setup.bash
ros2 launch frontier_exploration launch_auto_exploration.launch.py
```

## Terminal 3:
### Manual commands
Return to home and perform precision landing
```bash
ros2 service call /mission/return_home std_srvs/srv/Trigger
```

### Emergency
Land wherever it is
```bash
ros2 service call /mission/land_here std_srvs/srv/Trigger
```

