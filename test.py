import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import time
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel, AgentPair
from rl_agents import RLAgent


mdp = OvercookedGridworld.from_layout_name("cramped_room")
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)


agent_0 = GreedyHumanModel(mlam)
agent_0.set_agent_index(0)
agent_1 = GreedyHumanModel(mlam)
agent_1.set_agent_index(1)
agent_pair = AgentPair(agent_0, agent_1)

num_steps = 100000

for _ in range(num_steps):
    env.render()
    state = env.base_env.state
    #print(env.action_space.sample())
    #agent.action(state) is in format [most_likely_action, actions_probs]
    action_0 = agent_0.action(state)[0]
    print(action_0)
    action_1 = agent_1.action(state)[0]
    
    obs, reward, done, env_info = env.step((action_0, action_1))    
    print(obs)

