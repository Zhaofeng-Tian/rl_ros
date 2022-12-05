# Reinforcemen Learning with ROS/Gazebo Guidance 
## Build your python3 workspace
First, build python3 env using virtualenv

`virtualenv -p python3 your_env_name`<br>

then activate it to verify it is built successfully:

`source ~/your_env_name/bin/activate`<br>

Second, build your workspace compiling with python3 and meanwhile compile
the tf2_ros with python3. (tf2_ros was originally compiled with python2, 
so if you do not compile tf2_ros, it could cause issues)

`sudo apt update`<br>
`sudo apt install python3-catkin-pkg-modules python3-rospkg-modules python3-empy`<br>

prepare workspace:

`mkdir -p ~/your_ws/src; cd ~/your_ws`
`catkin_make`
`source devel/setup.bash`
`wstool init`
`wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5`
`wstool up`
`rosdep install --from-paths src --ignore-src -y -r`

compile workspace for Python3

`catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so`


## Download rl_ros package for python/Gazebo connection
Execute the following commands:<br>
`cd ~/your_ws/src`<br>
`https://github.com/Zhaofeng-Tian/rl_ros.git`<br>
`cd ~/your_ws`<br>
`catkin_make -j4`<br>
`source devel/setup.bash`<br>
`rosdep install rl_ros`<br>
