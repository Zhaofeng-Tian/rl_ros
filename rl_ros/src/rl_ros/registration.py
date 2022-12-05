#!/usr/bin/env python
import gym
import roslaunch
import rospy
import rospkg
import rosparam
import os
import sys
import subprocess
from gym.envs.registration import register
from gym import envs

def RegisterRL_Ros_Env(task_env, max_episode_steps=10000):
    """
    Registers all the ENVS supported in OpenAI ROS. This way we can load them
    with variable limits.
    Here is where you have to PLACE YOUR NEW TASK ENV, to be registered and accesible.
    return: False if the Task_Env wasnt registered, True if it was.
    """

    ###########################################################################
    # MovingCube Task-Robot Envs

    result = False
    if task_env == 'ZebratReal-v0':

        register(
            id=task_env,
            entry_point='rl_ros.task_envs.zebrat.zebrat_real_test:ZebratRealEnv',
            max_episode_steps=max_episode_steps,
        )
        # import our training environment
        result = True
        from rl_ros.envs.zebrat import zebrat_real_test    

    if task_env == 'ZebratReal-v1':

        register(
            id=task_env,
            entry_point='openai_ros.task_envs.zebrat.zebrat_discretized_real:ZebratRealEnv',
            max_episode_steps=max_episode_steps,
        )
        # import our training environment
        result = True
        from rl_ros.envs.zebrat import zebrat_discretized_real  


    if task_env == 'ZebratWall-v0':

        register(
            id=task_env,
            entry_point='rl_ros.envs.zebrat.zebrat_wall:ZebratWallEnv',
            max_episode_steps=max_episode_steps,
        )
        # import our training environment
        result = True
        from rl_ros.envs.zebrat import zebrat_wall

    if task_env == 'ZebratWall-v1':

        register(
            id=task_env,
            entry_point='rl_ros.envs.zebrat.zebrat_wall_continuous:ZebratWallContinuousEnv',
            max_episode_steps=max_episode_steps,
        )
        # import our training environment
        result = True
        from rl_ros.envs.zebrat import zebrat_wall_continuous



    if result:
        # We check that it was really registered
        supported_gym_envs = GetAllRegisteredGymEnvs()
        #print("REGISTERED GYM ENVS===>"+str(supported_gym_envs))
        assert (task_env in supported_gym_envs), "The Task_Robot_ENV given is not Registered ==>" + \
            str(task_env)

    return result

def GetAllRegisteredGymEnvs():
    """
    Returns a List of all the registered Envs in the system
    return EX: ['Copy-v0', 'RepeatCopy-v0', 'ReversedAddition-v0', ... ]
    """

    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    return env_ids

def StartRL_ROS_Environment(task_and_robot_environment_name):
    """
    It Does all the stuff that the user would have to do to make it simpler
    for the user.
    This means:
    0) Registers the TaskEnvironment wanted, if it exists in the Task_Envs.
    2) Checks that the workspace of the user has all that is needed for launching this.
    Which means that it will check that the robot spawn launch is there and the worls spawn is there.
    4) Launches the world launch and the robot spawn.
    5) It will import the Gym Env and Make it.
    """
    rospy.logwarn("Env: {} will be imported".format(
        task_and_robot_environment_name))
    result = RegisterRL_Ros_Env(task_env=task_and_robot_environment_name,
                                    max_episode_steps=10000)

    if result:
        rospy.logwarn("Register of Task Env went OK, lets make the env..."+str(task_and_robot_environment_name))
        env = gym.make(task_and_robot_environment_name)
    else:
        rospy.logwarn("Something Went wrong in the register")
        env = None

    return env


class ROSLauncher(object):
    def __init__(self, rospackage_name, launch_file_name, ros_ws_abspath="/home/tian/simulation_ws"):

        self._rospackage_name = rospackage_name
        self._launch_file_name = launch_file_name

        self.rospack = rospkg.RosPack()

        # Check Package Exists
        try:
            pkg_path = self.rospack.get_path(rospackage_name)
            rospy.logdebug("Package FOUND...")
        except rospkg.common.ResourceNotFound:
            rospy.logwarn("Package NOT FOUND, please check it out!")


        # Now we check that the Package path is inside the ros_ws_abspath
        # This is to force the system to have the packages in that ws, and not in another.
        if ros_ws_abspath in pkg_path:
            rospy.logdebug("Package FOUND in the correct WS!")
        else:
            rospy.logwarn("Package FOUND in "+pkg_path +
                          ", BUT not in the ws="+ros_ws_abspath+", lets Download it...")
            #pkg_path = self.DownloadRepo(package_name=rospackage_name, ros_ws_abspath=ros_ws_abspath)
                                         

        # If the package was found then we launch
        if pkg_path:
            rospy.loginfo(
                ">>>>>>>>>>Package found in workspace-->"+str(pkg_path))
            launch_dir = os.path.join(pkg_path, "launch")
            path_launch_file_name = os.path.join(launch_dir, launch_file_name)

            rospy.logwarn("path_launch_file_name=="+str(path_launch_file_name))

            source_env_command = "source "+ros_ws_abspath+"/devel/setup.bash;"
            roslaunch_command = "roslaunch  {0} {1}".format(rospackage_name, launch_file_name)
            command = source_env_command+roslaunch_command
            rospy.logwarn("Launching command="+str(command))

            p = subprocess.Popen(command, shell=True)

            state = p.poll()
            if state is None:
                rospy.loginfo("process is running fine")
            elif state < 0:
                rospy.loginfo("Process terminated with error")
            elif state > 0:
                rospy.loginfo("Process terminated without error")
            """
            self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(self.uuid)
            self.launch = roslaunch.parent.ROSLaunchParent(
                self.uuid, [path_launch_file_name])
            self.launch.start()
            """


            rospy.loginfo(">>>>>>>>>STARTED Roslaunch-->" +
                          str(self._launch_file_name))
        else:
            assert False, "No Package Path was found for ROS apckage ==>" + \
                str(rospackage_name)


def LoadYamlFileParamsTest(rospackage_name, rel_path_from_package_to_file, yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file) 
    path_config_file = os.path.join(config_dir, yaml_file_name)
    
    paramlist=rosparam.load_file(path_config_file)
    
    for params, ns in paramlist:
        rosparam.upload_params(ns,params)