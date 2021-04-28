import learn
import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from rl_agents import CentralizedAgentPair, DecentralizedAgent, StayRLAgent

mdp = OvercookedGridworld.from_layout_name("bottleneck")
#Other potentially interesting layouts: forced_coordination
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

rl_agent_1 = DecentralizedAgent()
rl_agent_2 = StayRLAgent()
central_agent = CentralizedAgentPair(rl_agent_1, rl_agent_2)

pair, reward = learn.run_episodes(central_agent, env, 500, 1000, render=True)
pair, reward = learn.run_episodes(pair, env, 10, 500, True)

