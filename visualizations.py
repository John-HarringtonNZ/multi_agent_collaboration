import matplotlib.pyplot as plt
import numpy as np

def get_num_q_vals(qvals_a0, qvals_a1, figure_title='testing_agent_qs'):

    fig, ax = plt.subplots()
    ax.plot(qvals_a0, label="Agent 0")
    ax.plot(qvals_a1, label="Agent 1")

    ax.set_xlabel('Episodes')
    ax.set_ylabel('Number of Q Values')
    ax.set_title('Number of Q Values over time for Decentralized Agent Pair')
    plt.legend(loc='upper right')
    plt.savefig(f'{figure_title}.png')

    return fig

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
