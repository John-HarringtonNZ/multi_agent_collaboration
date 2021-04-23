import learn
import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import RandomAgent
from rl_agents import DecentralizedAgent, RLAgent

mdp = OvercookedGridworld.from_layout_name("4100_handoff")
#Other potentially interesting layouts: forced_coordination
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

agent_rand = RandomAgent(all_actions=True)
rl_agent_1 = RLAgent()
rl_agent_2 = RLAgent()
central_agent = DecentralizedAgent(rl_agent_1, rl_agent_2)

pair, reward = learn.run_episodes(central_agent, env, 10000, 500, False)

pair.a0.epsilon = 0
pair.a1.epsilon = 0

pair, reward = learn.run_episodes(pair, env, 10, 500, True)
