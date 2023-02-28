#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
import numpy as np
import rospy


class Agent:
    def __init__(self, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=2, max_size=200000, tau=0.005,
                 obs_shape = 40, batch_size=64, noise=0.1, is_random = False, epsilon = 0.95,
                 chkpt_dir='/home/tian/simulation_ws/src/rl_ros/src/rl_ros/algorithms/DDPG/plots/models/',
                 load_path = '/home/tian/simulation_ws/src/rl_ros/src/rl_ros/algorithms/DDPG/plots/good_models/model1/',
                 mem_path = '/home/tian/simulation_ws/src/rl_ros/src/rl_ros/algorithms/DDPG/plots/memory/'):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, obs_shape, n_actions,mem_path,mem_ctr=0)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.noise = noise
        # self.max_action = env.action_space.high[0]
        # self.min_action = env.action_space.low[0]
        self.max_action = [0.6,0.6]
        self.min_action = [-0.6,-0.6]
        self.is_random = is_random
        self.epsilon = epsilon
        self.chkpt_dir = chkpt_dir
        self.load_path = load_path
        

        self.actor = ActorNetwork()
        # self.actor.model.build(10)
        self.critic = CriticNetwork()
        self.target_actor = ActorNetwork()
        self.target_critic = CriticNetwork()

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.actor_loss = 0
        self.critic_loss = 0
        self.policy = 'policy'

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir+'actor')
        self.target_actor.save(self.chkpt_dir+'target_actor')
        self.critic.save(self.chkpt_dir+'critic')
        self.target_critic.save(self.chkpt_dir+'target_critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.load_path+'actor')
        self.target_actor = \
            keras.models.load_model(self.load_path+'target_actor')
        self.critic = keras.models.load_model(self.load_path+'critic')
        self.target_critic = \
            keras.models.load_model(self.load_path+'target_critic')

    def choose_action(self, observation, evaluate=False, is_random=False):
        if evaluate:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            actions = self.actor(state) 
            actions += tf.random.normal(shape=[self.n_actions],mean=0.0, stddev=self.noise) 
            actions = tf.clip_by_value(actions, self.min_action, self.max_action)
            self.policy = 'DDPG with model loaded'
            return actions.numpy()[0]          
        else:
            if not is_random:
                state = tf.convert_to_tensor([observation], dtype=tf.float32)
                actions = self.actor(state)
                
                actions += tf.random.normal(shape=[self.n_actions],mean=0.0, stddev=self.noise)
                    #actions += np.random.normal(0.0,self.noise,self.n_actions)
                # note that if the env has an action > 1, we have to multiply by
                # max action at some point
                actions = tf.clip_by_value(actions, self.min_action, self.max_action)
                #actions = np.clip(actions, self.min_action, self.max_action)
                self.policy = 'DDPG'

                return actions.numpy()[0]
            else:
                actions = np.random.uniform(self.min_action, self.max_action, self.n_actions)+ np.random.normal(0.0,self.noise,self.n_actions)
                actions = np.clip(actions, self.min_action, self.max_action)
                self.policy = 'RANDOM'
                return actions

    def learn(self):
        print ("Memorey Index: " +str(self.memory.mem_cntr))
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            
            critic_value_ = tf.squeeze(self.target_critic((states_, target_actions)), 1)
            #print ("In Learning: Target critic: " +str(critic_value_))                
            
            critic_value = tf.squeeze(self.critic((states, actions)), 1)
            #print("In Learning: critic: "+str(critic_value))

            target = rewards + self.gamma*critic_value_*(1-done)
            #print ("In Learning: target value: "+str(target))
            critic_loss = keras.losses.MSE(target, critic_value)
            #critic_loss = tf.math.reduce_mean(tf.math.abs(target - critic_value))
            self.critic_loss = critic_loss.numpy()
            print ("***********  Critic LOSS :" + str(critic_loss.numpy()) + "  ******************")
        params = self.critic.trainable_variables
        # print("The shape of parameters: " + str(params))
        grads = tape.gradient(critic_loss, params)
        # print("The gradients: " + str(grads))
        # print("Zip(grads, params):  " + str(zip(grads, params)))
        self.critic.optimizer.apply_gradients(zip(grads, params))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic((states, new_policy_actions))
            actor_loss = tf.math.reduce_mean(actor_loss)
            self.actor_loss = actor_loss.numpy()
            print ("***********  Actor LOSS :" + str(actor_loss.numpy()) + "  ******************")
        params = self.actor.trainable_variables
        grads = tape.gradient(actor_loss, params)
        self.actor.optimizer.apply_gradients(zip(grads, params))

        self.update_network_parameters()
    
    def get_actor_loss(self):
        return self.actor_loss
    def get_critic_loss(self):
        return self.critic_loss