#!/usr/bin/env python3
import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, manage_memory, save_loss, save_score
import random

from rl_ros.registration import StartRL_ROS_Environment
import rospy
import rospkg
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    manage_memory()
    #env = gym.make('InvertedPendulumBulletEnv-v0')
    ENV_NAME = 'ZebratWall-v1'
    rospy.init_node('zebrat_sac', anonymous=True, log_level=rospy.WARN)      
    env = StartRL_ROS_Environment(ENV_NAME)
    pack_file = '/home/tian/simulation_ws/src/rl_ros/src/rl_ros/algorithms/SAC/plots/'

    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0],
                  chkpt_dir=pack_file+'model')
    n_games = 1000
    
    best_score = env.reward_range[0]

    score_list, reward_list, step_list, episode_list = [], [], [], []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    avg_score = 0
    n_steps = 0
    for i in range(n_games):
        ####
        episode_list.append(i+1)
        ####
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            reward_list.append(reward)
            print("Action: " +str(action))
            print("1. old obs, in shape" + str(observation))
            print("2. action chosen is: "+str(action))
            print("3. obs got, in shape: " +str(observation_))
            print("4. The reward for this step is :  " + str(reward))
            print("5. done flag: " + str(done))
            print(info)
            print("Is crash?: "+ env.crash)
            n_steps += 1
            step_list.append(n_steps)
            print('This is EPISODE: ' + str(i+1))
            print ('This is STEP: ' + str(n_steps))
            score += reward

            agent.store_transition(observation, action, reward,
                                   observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_list.append(score)
        avg_score = np.mean(score_list[-100:])

        if score > best_score:
            best_score = score
            if not load_checkpoint:
                agent.save_models()
        
        save_score(episode_list, score_list, pack_file + 'score.txt')     
        save_loss(step_list,reward_list, pack_file + 'loss.txt')

        print('episode {} score {:.1f} avg_score {:.1f}'.
              format(i, score, avg_score))

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_list, pack_file+'sac.png')


'''if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    ENV_NAME = 'ZebratWall-v1'
    rospy.init_node('zebrat_dqn', anonymous=True, log_level=rospy.WARN)      
    env = StartOpenAI_ROS_Environment(ENV_NAME)

    load_checkpoint = False
    score_list, reward_list, step_list, episode_list = [], [], [], []
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.shape[0], env = env, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_episodes = 20

    figure_file = '/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/PPO/plots/ppo.png'

    best_score = env.reward_range[0]


    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_episodes):
        episode_list.append(i+1)
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            reward_list.append(reward)
            print("Action: " +str(action))
            print("1. old obs, in shape" + str(observation))
            print("2. action chosen is: "+str(action))
            print("3. obs got, in shape: " +str(observation_))
            print("4. The reward for this step is :  " + str(reward))
            print("5. done flag: " + str(done))
            print(info)
            print("Is crash?: "+ env.crash)
            n_steps += 1
            print('This is EPISODE: ' + str(i+1))
            print ('This is STEP: ' + str(n_steps))
            score += reward
            agent.store_transition(observation, action,
                                   prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation
            step_list.append(n_steps)
        score_list.append(score)
        
        avg_score = np.mean(score_list[-100:])

        if (score > best_score) and (n_steps > 100):
            if not load_checkpoint:
                agent.save_models()
            best_score = score
        save_score(episode_list, score_list, '/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/PPO/plots/score.txt')     
        save_loss(step_list,reward_list,'/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/PPO/plots/loss.txt')

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_list))]
    plot_learning_curve(x, score_list, figure_file)'''