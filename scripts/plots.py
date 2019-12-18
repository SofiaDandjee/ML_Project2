# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_data(ratings):
    """plot the statistics result on raw rating data."""
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten() #flatten() returns a copy of the array collapsed into one dimension
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1] #sort() returns a sorted copy of an array
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    # plot
    plt.rcParams["figure.figsize"] = (7,4)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_movies_per_user, color='#D95319')
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_users_per_movie, color='#A2142F')
    ax2.set_xlabel("Movies")
    ax2.set_ylabel("Number of ratings (sorted)")
    ax2.grid()
    
    plt.tight_layout()
    plt.savefig("stat_ratings")
    plt.show()
    return num_items_per_user, num_users_per_item

def plot_train_test_data(train):
    """visualize the train data."""
    plt.rcParams["figure.figsize"] = (2,100)
    plt.spy(train, precision=0.01, markersize=0.12, color='#7E2F8E') #plot the sparsity pattern of a 2D array
    plt.xlabel("Movies")
    plt.ylabel("Users")
    plt.title("Training data")
    plt.savefig("sparsity")
    plt.show()