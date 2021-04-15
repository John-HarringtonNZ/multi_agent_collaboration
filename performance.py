import learn, random

import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent, AgentPair
from rl_agents import DecentralizedAgent, RLAgent, ApproximateQAgent



def get_avg_rewards(a_pair):
    _, rewards1 = learn.run_episodes_arr(a_pair, env, 1000, 1000, 123, False)
    _, rewards2 = learn.run_episodes_arr(a_pair, env, 1000, 1000, 456, False)
    _, rewards3 = learn.run_episodes_arr(a_pair, env, 1000, 1000, 999, False)
    _, rewards4 = learn.run_episodes_arr(a_pair, env, 1000, 1000, 101010101, False)
    avged = [0]*len(rewards1)
    for i in range(len(rewards1)):
        avged[i] = (rewards1[i] + rewards2[i] + rewards3[i] + rewards4[i]) / 4.0
    return avged
###############################################


mdp = OvercookedGridworld.from_layout_name("cramped_room")
#Other potentially interesting layouts: forced_coordination
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)



results = {}

a1 = ApproximateQAgent(0, mdp, mlam)
a2 = ApproximateQAgent(1, mdp, mlam)
central_agent = DecentralizedAgent(a1, a2)
results['Approximate Q Agent', get_avg_rewards(central_agent)]


a1 = BasicCommunicateAgent()
a2 = BasicCommunicateAgent()
central_agent = CommunicationPair(a1, a2)
results['Approximate Q Agent', get_avg_rewards(central_agent)]

print(results)

