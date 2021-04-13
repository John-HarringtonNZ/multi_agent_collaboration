import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import time
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel, AgentPair
from rl_agents import RLAgent, CentralizedAgent


mdp = OvercookedGridworld.from_layout_name("cramped_room")
#Other potentially interesting layouts: forced_coordination
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)

#agent_0 = GreedyHumanModel(mlam)
#agent_1 = GreedyHumanModel(mlam)
#central_agent = CentralizedAgent(agent_0, agent_1)
central_agent = CentralizedAgent(RandomAgent(), RandomAgent())
central_agent.set_agent_index(0)

num_steps = 100000

for t in range(num_steps):
    env.render()
    state = env.base_env.state

    joint_action = central_agent.action(state)
    act1 = joint_action[0][0][0]
    act2 = joint_action[0][1][0]
    obs, reward, done, env_info = env.step((act1, act2))
    if done:
        print("Run finished after {} timesteps".format(t+1))
        break
    #time.sleep(0.2)

print("Run finished going through all the timesteps".format(num_steps))
env.close()
