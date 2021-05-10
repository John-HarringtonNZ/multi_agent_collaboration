import learn
import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import RandomAgent
from rl_agents import DecentralizedAgent, RLAgent, StayRLAgent
import numpy as np

#mdp = OvercookedGridworld.from_layout_name("4100_handoff")
mdp = OvercookedGridworld.from_layout_name("cramped_room")
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

custom_sparse_rewards = {
   'deliver_soup': 1000,
   'add_onion_to_pot': 100,
   'pickup_onion': 1,
   'add_soup_to_plate': 10000
}
mdp.set_sparse_rewards(custom_sparse_rewards)

agent_rand = RandomAgent(all_actions=True)
rl_agent_1 = RLAgent()
rl_agent_2 = StayRLAgent()
central_agent = DecentralizedAgent(rl_agent_1, rl_agent_2)

#decentralized agent will first deliver soup in episode 8 (500 steps each) seed=241
seed = np.random.randint(100000)
pair, reward, q_values = learn.run_episodes_arr_q(central_agent, env, num_episodes=1000, num_steps=500, seed=seed, render=True)

print("CUMULATIVE GAME STATS\n", base_env.game_stats)
pair.a0.epsilon = 0
pair.a1.epsilon = 0

get_ave_episode_rewards(rewards)

pair, reward = learn.run_episodes_arr(pair, env, 10, 500, render=True)
