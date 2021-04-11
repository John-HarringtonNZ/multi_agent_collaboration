import learn

import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from rl_agents import CentralizedAgent, RLAgent

mdp = OvercookedGridworld.from_layout_name("cramped_room")
#Other potentially interesting layouts: forced_coordination
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)

agent_rand = RandomAgent(all_actions=True)
rl_agent = RLAgent()
central_agent = CentralizedAgent(rl_agent, rl_agent, allow_duplicate_agents=True)

pair, reward = learn.run_episodes(central_agent, env, 1000, 1000, True)
