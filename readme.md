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

`mkdir -p ~/catkin_ws/src; cd ~/catkin_ws`


## Download rl_ros package for python/Gazebo connection
Execute the following commands:<br>
`cd ~/your_ws/src`<br>
`https://github.com/Zhaofeng-Tian/rl_ros.git`<br>
`cd ~/your_ws`<br>
`catkin_make -j4`<br>
`source devel/setup.bash`<br>
`rosdep install rl_ros`<br>
