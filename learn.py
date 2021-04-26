import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import time, random, numpy
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel, AgentPair

def run_game(agent_pair, env, num_steps, shouldUseIntermediateRewards=False, render=False):

    total_game_reward = 0

    for t in range(num_steps):
        if render:  
            env.render()
            #time.sleep(0.1)

        state = env.base_env.state

        #import pdb
        #pdb.set_trace()
        act1, act2 = agent_pair.joint_action(state)
        obs, reward, done, env_info = env.step((act1, act2))

        if shouldUseIntermediateRewards:
            reward_pair = [(env_info['sparse_r_by_agent'][0] + env_info['shaped_r_by_agent'][1]), (env_info['sparse_r_by_agent'][1] + env_info['shaped_r_by_agent'][1])]
        else:
            reward_pair = [(env_info['sparse_r_by_agent'][0]), (env_info['sparse_r_by_agent'][1])]

        nextState = env.base_env.state

        reward = reward_pair[0] + reward_pair[1]
        agent_pair.observeTransition(state, (act1, act2), nextState, reward_pair)
        total_game_reward += reward
        if reward > 0:
            print(f'Got {reward_pair[0]} {reward_pair[1]} reward!')
        if done:
            print("Run finished after {} timesteps".format(t+1))
            break

    #for q_val in agent_pair.a0.q_values.keys():
    #    print(q_val)
    #print(len(agent_pair.a0.q_values))

    return (agent_pair, total_game_reward)

def run_episodes(agent_pair, env, num_episodes, num_steps, seed=None, render=False):
    if seed:
        random.seed(seed)
        numpy.random.seed(seed)
    average_game_reward = 0
    total_episodes_reward = 0
    rewards = []
    for e in range(num_episodes):
        env.reset(regen_mdp=False)
        print(f"Starting episode {e}, Ave: {total_episodes_reward/(e+1)}")
        agent_pair, e_reward = run_game(agent_pair, env, num_steps, render=render)
        total_episodes_reward += e_reward
        rewards.append(e_reward)

    ave_reward = total_episodes_reward / num_episodes
    print(f"Average reward: {ave_reward}")
    print("Rewards:", rewards)
    return agent_pair, ave_reward


def run_episodes_arr(agent_pair, env, num_episodes=100, num_steps=100, seed=None, render=False):
    if seed:
        random.seed(seed)
        numpy.random.seed(seed)
    average_game_reward = 0
    total_episodes_reward = 0
    rewards = []
    for e in range(num_episodes):
        env.reset(regen_mdp=False)
        print(f"Starting episode {e}, Ave: {total_episodes_reward/(e+1)}")
        agent_pair, e_reward = run_game(agent_pair, env, num_steps, render=render)
        total_episodes_reward += e_reward
        rewards.append(e_reward)

    ave_reward = total_episodes_reward / num_episodes
    #print(f"Average reward: {ave_reward}")
    #print("Rewards:", rewards)
    return agent_pair, rewards

def run_episodes_arr_q(agent_pair, env, num_episodes, num_steps, seed=None, render=False):
    if seed:
        random.seed(seed)
        numpy.random.seed(seed)
    average_game_reward = 0
    total_episodes_reward = 0
    q_counts = []
    rewards = []
    for e in range(num_episodes):
        env.reset(regen_mdp=False)
        print(f"Starting episode {e}, Ave: {total_episodes_reward/(e+1)}")
        agent_pair, e_reward = run_game(agent_pair, env, num_steps, render=render)
        total_episodes_reward += e_reward
        rewards.append(e_reward)
        q_counts.append(agent_pair.getNumQVals())

    ave_reward = total_episodes_reward / num_episodes
    #print(f"Average reward: {ave_reward}")
    #print("Rewards:", rewards)
    return agent_pair, rewards, q_counts
