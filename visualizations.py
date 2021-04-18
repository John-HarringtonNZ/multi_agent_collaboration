import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

def get_ave_episode_rewards(ave_episode_dict, figure_title='testing_agents'):

    fig, ax = plt.subplots()

    for plot_title, rewards in ave_episode_dict.items():

        ax.plot(rewards, label=plot_title)

    ax.set_xlabel('Episodes')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward over time for Overcooked Agents')
    plt.legend(loc='upper right')
    plt.savefig(f'{figure_title}.png')

    return fig


def windowed_average_plot(ave_episode_dict, window_size=10, figure_title='testing_agents'):

    windowed = {}
    for plot_title, rewards in ave_episode_dict.items():

        window = np.ones(int(window_size))/float(window_size)
        windowed[plot_title] =  np.convolve(rewards, window, 'same')

    fig = get_ave_episode_rewards(windowed, figure_title)

    return fig 
