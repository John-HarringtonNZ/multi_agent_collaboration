import learn
import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import RandomAgent
from rl_agents import DecentralizedAgent, RLAgent
import numpy as np

mdp = OvercookedGridworld.from_layout_name("4100_handoff")
#Other potentially interesting layouts: forced_coordination
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

custom_sparse_rewards = {
    'deliver_soup': 10000,
    'add_onion_to_pot': 100,
    'pickup_onion': 1,
    'add_soup_to_plate': 1000
}
mdp.set_sparse_rewards(custom_sparse_rewards)

agent_rand = RandomAgent(all_actions=True)
rl_agent_1 = RLAgent()
rl_agent_2 = RLAgent()
central_agent = DecentralizedAgent(rl_agent_1, rl_agent_2)

np.random.seed(42)
pair, reward = learn.run_episodes_arr(central_agent, env, 5000, 100, False)

from visualizations import *
windowed_average_plot({'Decentralized_agent_with_intermediate_rewards': reward}, figure_title="results/decentralized_agent_with_intermediate_rewards")


pair.a0.epsilon = 0
pair.a1.epsilon = 0

#pair, reward = learn.run_episodes(pair, env, 10, 500, True)

