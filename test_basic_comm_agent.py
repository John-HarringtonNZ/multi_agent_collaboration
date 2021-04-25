import learn
import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent
from rl_agents import BasicCommunicateAgent, CommunicationPair

mdp = OvercookedGridworld.from_layout_name("4100_handoff")
#Other potentially interesting layouts: forced_coordination
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
custom_sparse_rewards = {
    'deliver_soup': 10000,
    'add_onion_to_pot': 100,
    'pickup_onion': 1
}
mdp.set_sparse_rewards(custom_sparse_rewards)

agent_rand = RandomAgent(all_actions=True)
comm_agent_1 = BasicCommunicateAgent()
comm_agent_2 = BasicCommunicateAgent()
central_agent = CommunicationPair(comm_agent_1, comm_agent_2)

pair, reward = learn.run_episodes(central_agent, env, 600, 100, False)

pair, reward = learn.run_episodes(pair, env, 10, 500, True)
