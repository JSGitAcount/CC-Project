# from moonlander_env import MoonlanderWorldEnv
import itertools
from pprint import pprint

import numpy as np
import cv2

# env = MoonlanderWorldEnv()
#
# observation, _ = env.reset()
# env.state = [-1, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, -1,
#              -1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, -1,
#              -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
#              -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
#              -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
#              -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
#              -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
#              -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
#              -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
#              -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, ]

# print(cv2.GaussianBlur(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]]).astype(
#     np.int16), (7, 7), 0, ))
#
# print(cv2.GaussianBlur(np.array([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, ],
#                                  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, ],
#                                  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, ],
#                                  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, ],
#                                  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, ],
#                                  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, ],
#                                  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, ],
#                                  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, ],
#                                  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, ],
#                                  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, ]]).astype(
#     np.int16), (7, 7), 0, ))
#
# print(cv2.GaussianBlur(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
#                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]]).astype(
#     np.int16), (7, 7), 0, ))
#
# print(cv2.GaussianBlur(np.array([[127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, ],
#                                  [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, ],
#                                  [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, ],
#                                  [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, ],
#                                  [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, ],
#                                  [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, ],
#                                  [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, ],
#                                  [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, ],
#                                  [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, ],
#                                  [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, ]]).astype(
#     np.int16), (7, 7), 127, ))
# # _, _, _, _, _ = env.step(0)

import csv
import pandas as pd
import matplotlib.pyplot as plt


def plot_histogram_dodge(filename: str, task_name: str):
    df = pd.read_csv(filename)
    df = df.stack()
    # we choose to clip 1% -> which is to -3
    counts = df.value_counts(normalize=True)
    print(counts.sort_index().cumsum())
    df = df.clip(-3, 10)

    hist = df.hist(bins=70)

    # Adding title and labels
    plt.title(f'Histogram for Rewards in {task_name} (n={df.shape[0]})')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')

    # Display the histogram
    plt.show()


def plot_histogram_collect(filename: str, task_name: str):
    df = pd.read_csv(filename)
    df = df.stack()
    # we choose to clip 1% -> which is to 62
    counts = df.value_counts(normalize=True)
    print(counts.sort_index().cumsum())
    df = df.clip(0, 62)

    hist = df.hist(bins=70)

    # Adding title and labels
    plt.title(f'Histogram for Rewards in {task_name} (n={df.shape[0]})')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')

    # Display the histogram
    plt.show()


dodge_filename = '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/src/data/bdf7da1/MoonlanderWorld-dodge-gaussian-v0/15-10-05/dodge_rewards.csv'
plot_histogram_dodge(filename=dodge_filename, task_name='Dodge Asteroids')
collect_filename = '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/src/data/bdf7da1/MoonlanderWorld-collect-gaussian-v0/15-13-41/collect_rewards.csv'
plot_histogram_collect(filename=collect_filename, task_name='Collect Asteroids')
