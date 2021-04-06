import gym
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, ObjectState, PlayerState, OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import time

mdp = OvercookedGridworld.from_layout_name("cramped_room")
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.featurize_state_mdp, display=True)

from overcooked_ai_py.agents.agent import RandomAgent, StayAgent, AgentPair

agent_0 = RandomAgent(all_actions=True)
agent_1 = RandomAgent(all_actions=True)
#ap = AgentPair(a0, a1)

#env.base_env.run_agents(ap, include_final_state=True)
ob = env.reset()

for _ in range(100000):
    env.render()
    state = env.base_env.state
    print(env.action_space.sample())
    action_0 = agent_0.action(state)[0]
    action_1 = agent_1.action(state)[0]
    print(action_0)
    obs, reward, done, env_info = env.step((action_0, action_1))    
    time.sleep(0.25)

