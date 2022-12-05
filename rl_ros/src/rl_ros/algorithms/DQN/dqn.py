#!/usr/bin/env python3
"""
Author: Zhaofeng Tian
Email: shoguntian@gmail.com
"""
from pickle import FALSE, TRUE
import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, manage_memory, save_loss, save_score
import random

from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import rospy
import rospkg
import tensorflow as tf

if __name__ == '__main__':
    manage_memory()
    ENV_NAME = 'ZebratWall-v0'
    #ENV_NAME = 'ZebratReal-v1'
    rospy.init_node('zebrat_dqn', anonymous=True, log_level=rospy.WARN)      
    env = StartOpenAI_ROS_Environment(ENV_NAME)
    agent = Agent(gamma=0.99, epsilon=0.5, lr=0.0001,
                  input_dims=(env.observation_space.shape),
                  n_actions=env.action_space.n, mem_size=50000, eps_min=0.05,
                  batch_size=32, replace=1000, eps_dec=8e-6,
                  chkpt_dir='/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/DQN/models/')
    best_score = -np.inf
    load_checkpoint = False
    #load_train = False
    n_games = 503

    if load_checkpoint:
        agent.load_models()
        agent.epsilon = agent.eps_min

    figure_file = '/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/DQN/plots/dqn.png'

    n_steps = 0
    score_list, reward_list, step_list, episode_list = [], [], [], []

    for i in range(n_games):
        episode_list.append(i+1)
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            print("Action: " +str(action))
            observation_, reward, done, info = env.step(action)
            reward_list.append(reward)
            print("1. old obs, in shape" + str(observation))
            print("2. action chosen is: "+str(action))
            print("3. obs got, in shape: " +str(observation_))
            print("4. The reward for this step is :  " + str(reward))
            print("5. done flag: " + str(done))
            print('Step: ' + str(n_steps))
            print('Episode: ' + str(i+1))
            print(info)
            print("Is crash?: "+ env.crash)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                       reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
            step_list.append(n_steps)
        score_list.append(score)
        

        avg_score = np.mean(score_list[-100:])
        print('episode {} score {:.1f} avg score {:.1f} '
              'best score {:.1f} epsilon {:.2f} steps {}'.
              format(i, score, avg_score, best_score, agent.epsilon,
                     n_steps))

        if (score > best_score) and (n_steps > 100):
            if not load_checkpoint:
                agent.save_models()
            best_score = score
        save_score(episode_list, score_list, '/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/DQN/plots/score.txt')
        save_loss(step_list,reward_list,'/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/DQN/plots/loss.txt')

    x = [i+1 for i in range(len(score_list))]
