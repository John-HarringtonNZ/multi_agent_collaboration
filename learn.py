import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import time, random, numpy
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel, AgentPair
import math

def run_game(agent_pair, env, num_steps, shouldUseIntermediateRewards=False, render=False):

    total_game_reward = 0

    for t in range(num_steps):
        if render:  
            env.render()

        state = env.base_env.state
        act1, act2 = agent_pair.joint_action(state)
        obs, reward, done, env_info = env.step((act1, act2))

        if shouldUseIntermediateRewards:
            reward_pair = [(env_info['sparse_r_by_agent'][0] + env_info['shaped_r_by_agent'][1]), (env_info['sparse_r_by_agent'][0] + env_info['shaped_r_by_agent'][1])]
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

    return (agent_pair, total_game_reward)

def run_episodes(agent_pair, env, num_episodes, num_steps, seed=None, render=False, q_dec_rate=1):
    agent_pair, rewards = run_episodes_arr(agent_pair, env, num_episodes, num_steps, seed, render, q_dec_rate)
    return agent_pair, sum(rewards)/len(rewards)

def run_episodes_arr(agent_pair, env, num_episodes=100, num_steps=100, seed=None, render=False, q_dec_rate=1):
    agent_pair, rewards, _ = run_episodes_arr_q(agent_pair, env, num_episodes, num_steps, seed, render, q_dec_rate)
    return agent_pair, rewards

def run_episodes_arr_q(agent_pair, env, num_episodes, num_steps, seed=None, render=True, q_dec_rate=1):
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
        agent_pair.decay_epsilon(q_dec_rate)

    ave_reward = total_episodes_reward / num_episodes
    return agent_pair, rewards, q_counts


def run_episodes_arr_q_differential(agent_pair, env, num_episodes, num_steps, seed=None, render=True, q_delta_rate=0.1):
    
    if seed:
        random.seed(seed)
        numpy.random.seed(seed)
    average_game_reward = 0
    total_episodes_reward = 0
    q_counts = []
    rewards = []

    #Determine how positive/negative the last 10 episodes has been
    def rewards_differential(past_rewards):
        if len(past_rewards) < 10:
            return "None"
        recent_past = past_rewards[-10:]
        sum_1 = sum(recent_past[:5])
        sum_2 = sum(recent_past[5:])

        return abs((sum_2 - sum_1) / (sum_1 + sum_2))

    def update_epsilon():
        rewards_diff = rewards_differential(rewards)
        if rewards_diff == "None":
            return

        if rewards_diff > 500:
            f = 0.1
        else:
            f = 4* (1.0 / (1.0 + math.exp(rewards_diff))) - 1
        agent_pair.a0.epsilon = min(max(0, agent_pair.a0.epsilon + 0.01*f), 1)
        agent_pair.a1.epsilon = min(max(0, agent_pair.a1.epsilon + 0.01*f), 1)

    for e in range(num_episodes):
        env.reset(regen_mdp=False)
        print(f"Starting episode {e}, Ave: {total_episodes_reward/(e+1)}")
        agent_pair, e_reward = run_game(agent_pair, env, num_steps, render=render)
        total_episodes_reward += e_reward
        rewards.append(e_reward)
        q_counts.append(agent_pair.getNumQVals())
        update_epsilon()

    ave_reward = total_episodes_reward / num_episodes
    return agent_pair, rewards, q_counts
