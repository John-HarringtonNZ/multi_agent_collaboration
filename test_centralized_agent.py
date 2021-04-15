import learn

import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent, AgentPair, StayAgent
from rl_agents import DecentralizedAgent, RLAgent, ApproximateQAgent, CentralizedAgentPair, CentralAgent

mdp = OvercookedGridworld.from_layout_name("cramped_room")
#Other potentially interesting layouts: forced_coordination
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)

agent_rand = RandomAgent(all_actions=True)
rl_agent_1 = CentralAgent()
rl_agent_2 = StayAgent()
central_agent = CentralizedAgentPair(rl_agent_1, rl_agent_2)

pair, reward = learn.run_episodes(central_agent, env, 1000, 1000, False)

pair, reward = learn.run_episodes(pair, env, 10, 500, True)
