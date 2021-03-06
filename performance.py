import learn, random

import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from rl_agents import *
from operator import add
from overcooked_ai_py.agents.agent import StayAgent

def get_avg_rewards(a_pair, num_episodes, num_steps = 1000, seeds=[123,456]):
    rewards = []
    for s in seeds:
        a_pair.a0.reset_q_values()
        a_pair.a1.reset_q_values()

        _, rew = learn.run_episodes_arr(a_pair, env, num_episodes, num_steps, s, False)
        rewards.append(rew)

    total_reward = [sum(x) for x in zip(*rewards)]
    ave_reward = [x/len(seeds) for x in total_reward]

    return ave_reward
###############################################

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
a2 = RLAgent()
decentral_agent = DecentralizedAgent(a1, a2)
agent_names['Decentralized Agent'] = decentral_agent

a1 = BasicCommunicateAgent()
a2 = BasicCommunicateAgent()
basic_comm_agent = CommunicationPair(a1, a2)
agent_names['Communication Agent'] = basic_comm_agent

rl_agent_1 = CentralAgent()
rl_agent_2 = StayRLAgent()
central_agent = CentralizedAgentPair(rl_agent_1, rl_agent_2)
agent_names['Centralized Agent'] = central_agent

a1 = ApproximateQAgent(0, mlam)
a2 = ApproximateQAgent(1, mlam)
approx_agent = DecentralizedAgent(a1, a2)
approx_agent.set_mdp(mdp)
agent_names['Approximate Q Agent'] = approx_agent

results = {}
num_episodes = 100
for name, agent in agent_names.items():
    results[name] = get_avg_rewards(agent, num_episodes, num_steps=100)

from visualizations import *
windowed_average_plot(results, figure_title=f'results/agent_comparison_with_intermediate_rewards_without_drop_{num_steps}_steps')