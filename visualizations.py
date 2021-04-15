import datetime as dt
import matplotlib.pyplot as plt


def get_ave_episode_rewards(ave_episode_dict):

    fig, ax = plt.subplots()

    for plot_title, rewards in ave_episode_dict.items():

        ax.plot(rewards, label=plot_title)

    ax.set_xlabel('Episodes')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward over time for Overcooked Agents')

    return fig

test_dict = {'central':[0,1,1,1,0,10], 'decentral':[0,0,1,1,1,2,3]}

get_ave_episode_rewards(test_dict)

plt.show()