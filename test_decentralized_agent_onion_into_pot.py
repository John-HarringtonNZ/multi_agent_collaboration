import learn
import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import RandomAgent
from rl_agents import DecentralizedAgent, RLAgent

#mdp = OvercookedGridworld.from_layout_name("4100_handoff")
mdp = OvercookedGridworld.from_layout_name("cramped_room")
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

custom_sparse_rewards = {
    'deliver_soup': 0,
    'add_onion_to_pot': 30,
    'pickup_onion': 0
}
mdp.set_sparse_rewards(custom_sparse_rewards)

agent_rand = RandomAgent(all_actions=True)
rl_agent_1 = RLAgent()
rl_agent_2 = RLAgent()
central_agent = DecentralizedAgent(rl_agent_1, rl_agent_2)

#decentralized agent will first deliver soup in episode 8 (500 steps each) seed=241
pair, reward = learn.run_episodes_arr(central_agent, env, 200, 500, 241, render=False)

pair.a0.epsilon = 0
pair.a1.epsilon = 0

pair, reward = learn.run_episodes(pair, env, 10, 500, True)
