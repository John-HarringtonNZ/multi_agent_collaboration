import learn
import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from rl_agents import *
from operator import add
from overcooked_ai_py.agents.agent import StayAgent
import numpy as np

def get_avg_rewards(a_pair, num_episodes, num_steps = 1000, seeds=[123,456,999,101010101]):
    rewards = []
    for s in seeds:
        a_pair.a0.reset_q_values()
        a_pair.a1.reset_q_values()

        _, rew = learn.run_episodes_arr(a_pair, env, num_episodes, num_steps, s, render=False)
        rewards.append(rew)

    total_reward = [sum(x) for x in zip(*rewards)]
    ave_reward = [x/len(seeds) for x in total_reward]

    return ave_reward

# ONLY FOR DECENTRALIZED AGENT
def get_avg_rewards_and_qs(decentralized_agent_pair, num_episodes, num_steps = 1000, seeds=None):#,456,999,101010101]):
    if not seeds:
        seeds = [np.random.randint(10000)]
    rewards = []
    q_val_counts_a0 = []
    q_val_counts_a1 = []
    for s in seeds:
        decentralized_agent_pair.a0.reset_q_values()
        decentralized_agent_pair.a1.reset_q_values()

        _, rew, q_counts = learn.run_episodes_arr_q(decentralized_agent_pair, env, num_episodes, num_steps, s, render=False)
        q_val_counts_a0 = [x[0] for x in q_counts]
        q_val_counts_a1 = [x[1] for x in q_counts]
        rewards.append(rew)

    total_reward = [sum(x) for x in zip(*rewards)]
    ave_reward = [x/len(seeds) for x in total_reward]

    return ave_reward, (q_val_counts_a0, q_val_counts_a1)
###############################################

#mdp = OvercookedGridworld.from_layout_name("4100_isolated")
mdp = OvercookedGridworld.from_layout_name("4100_isolated")
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)
mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
agent_names = {}

custom_sparse_rewards = {
    'deliver_soup': 10000,
    'add_onion_to_pot': 100,
    'pickup_onion': 1,
    'add_soup_to_plate': 1000
}
mdp.set_sparse_rewards(custom_sparse_rewards)

a1 = RLAgent()
a2 = StayRLAgent()
decentral_agent = DecentralizedAgent(a1, a2)
agent_names['Decentralized Agent'] = decentral_agent

results = {}
num_episodes = 5000
results['decentralized Agent'], q_val_counts = get_avg_rewards_and_qs(decentral_agent, num_episodes, num_steps=300)

q_counts_a0 = q_val_counts[0]
q_counts_a1 = q_val_counts[1]

from visualizations import *
windowed_average_plot(results, figure_title=f'decentralized_single_agent_{num_episodes}_{num_steps}')
get_num_q_vals(q_counts_a0, q_counts_a1, figure_title='decentral_single_agent_q_vals')