import learn, random

import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from rl_agents import *
from operator import add
from overcooked_ai_py.agents.agent import StayAgent

def get_avg_rewards(a_pair, num_episodes):
    seeds = [123,456,999,101010101]
    rewards = []
    for s in seeds:
        a_pair.a0.reset_q_values()
        a_pair.a1.reset_q_values()

        _, rew = learn.run_episodes_arr(a_pair, env, num_episodes, 1000, s, False)
        rewards.append(rew)

    total_reward = [sum(x) for x in zip(*rewards)]
    ave_reward = [x/len(seeds) for x in total_reward]
    print(ave_reward)
    return ave_reward
###############################################


mdp = OvercookedGridworld.from_layout_name("cramped_room")
#Other potentially interesting layouts: forced_coordination
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
num_episodes = 1000
results = {}


print('Testing Decentralized Q agent\n\n')
a1 = RLAgent()
a2 = RLAgent()
central_agent = DecentralizedAgent(a1, a2)
results['Decentralized Agent'] =  get_avg_rewards(central_agent, num_episodes)

print('Testing communication agent\n\n')
a1 = BasicCommunicateAgent()
a2 = BasicCommunicateAgent()
central_agent = CommunicationPair(a1, a2)
results['Communication Agent'] = get_avg_rewards(central_agent, num_episodes)

print('Testing Centralized agent\n\n')
rl_agent_1 = CentralAgent()
rl_agent_2 = StayRLAgent()
central_agent = CentralizedAgentPair(rl_agent_1, rl_agent_2)
results['Centralized Agent'] = get_avg_rewards(central_agent, num_episodes)

print('Testing approximate Q agent\n\n')
a1 = ApproximateQAgent(0, mdp, mlam)
a2 = ApproximateQAgent(1, mdp, mlam)
central_agent = DecentralizedAgent(a1, a2)
results['Approximate Q Agent'] =  get_avg_rewards(central_agent, num_episodes)


from visualizations import get_ave_episode_rewards

get_ave_episode_rewards(results)