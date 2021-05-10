import learn
import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair
from rl_agents import CentralizedAgentPair, CentralAgent, StayRLAgent

mdp = OvercookedGridworld.from_layout_name("cramped_room")
#Other potentially interesting layouts: forced_coordination
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.mdp.flatten_state, display=True)

custom_sparse_rewards = {
   'deliver_soup': 1000,
   'add_onion_to_pot': 100,
   'pickup_onion': 1,
   'add_soup_to_plate': 10000
}
mdp.set_sparse_rewards(custom_sparse_rewards)

rl_agent_1 = CentralAgent()
rl_agent_2 = StayRLAgent()
central_agent = CentralizedAgentPair(rl_agent_1, rl_agent_2)

pair, reward = learn.run_episodes(central_agent, env, 5000, 100, render=True)
pair, reward = learn.run_episodes(pair, env, 10, 500, render=True)

