#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def manage_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

# save step, loss, episodes, average score, best score
def save_loss(step_list, reward_list, save_path):
    if os.path.isfile(save_path):
        os.remove(save_path)
    with open(save_path, "a") as f:
        for i in range(len(step_list)):
            f.write('{:}   {:} \n'. format(step_list[i], reward_list[i]))
            
def save_score(episode_list, score_list, save_path):
    if os.path.isfile(save_path):
        os.remove(save_path)
    with open(save_path, "a") as f:
        for i in range(len(episode_list)):
            f.write('{:}  {:} \n '. format(episode_list[i],score_list[i]))