import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import time
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel, AgentPair

def run_game(agent_pair, env, num_steps, render=False, visualize=False):

    total_game_reward = 0

    for t in range(num_steps):
        if render:  
            env.render()
            time.sleep(0.1)
        
        state = env.base_env.state

        act1, act2 = agent_pair.joint_action(state)
        obs, reward, done, env_info = env.step((act1, act2))
        nextState = env.base_env.state

        print(nextState)
        print()
        agent_pair.observeTransition(state, (act1, act2), nextState, reward)
        total_game_reward += reward
        if reward > 0:
            print(f'Got {reward} reward!')
        if done:
            print("Run finished after {} timesteps".format(t+1))
            break
    
    #for q_val in agent_pair.a0.q_values.keys():
    #    print(q_val)
    #print(len(agent_pair.a0.q_values))

    return (agent_pair, total_game_reward)

def run_episodes(agent_pair, env, num_episodes, num_steps, render=False):

    average_game_reward = 0
    total_episodes_reward = 0
    for e in range(num_episodes):
        env.reset()
        print(f"Starting episode {e}, Ave: {total_episodes_reward/(e+1)}")
        agent_pair, e_reward = run_game(agent_pair, env, num_steps, render)
        total_episodes_reward += e_reward
    
    ave_reward = total_episodes_reward / num_episodes
    print(f"Average reward: {ave_reward}")
    return agent_pair, ave_reward
