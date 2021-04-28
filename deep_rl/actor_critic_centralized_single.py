import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Create Overcooked Environment:
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair, StayAgent

mdp = OvercookedGridworld.from_layout_name("cramped_room")
#Other potentially interesting layouts: forced_coordination
base_env = OvercookedEnv.from_mdp(mdp)
env = gym.make('Overcooked-v0')
env.custom_init(base_env, base_env.mdp.flatten_state, display=True)
input_size = env.featurize_fn(env.base_env.state)[0].shape

# Configuration parameters for the whole setup
#seed = 42
#np.random.seed(seed)
#tf.random.set_seed(seed)
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 200
#env = gym.make("CartPole-v0")  # Create the environment
#env.seed(seed)
#State is numpy.ndarray
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_single_actions = 6
num_inputs = input_size[0]
num_actions = num_single_actions ** 1
num_hidden = 40

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
common_2 = layers.Dense(num_hidden, activation="relu")(common)
common_3 = layers.Dense(num_hidden, activation="relu")(common_2)
action = layers.Dense(num_actions, activation="softmax")(common_3)
critic = layers.Dense(30)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])


optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

#Add shape function to modify rewards
custom_sparse_rewards = {
    'deliver_soup': 1000,
    'add_onion_to_pot': 100,
    'pickup_onion': 1
}
mdp.set_sparse_rewards(custom_sparse_rewards)


while True:  # Run until solved
    state, _ = env.reset(regen_mdp=False, return_only_state=True)

    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            #print(f"{timestep}/{max_steps_per_episode}")
            env.render() 
            #; Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            print(action_probs)
            #print(critic_value)

            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            #print("Action:")
            #print(action)
            #action_1, action_2 = action // num_single_actions, action % num_single_actions
            action_1 = action
            action_2 = 4

            #print((action_1, action_2))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            _, reward, done, _ = env.step((action_1, action_2), action_as_ind=True)

            if reward > 0:
                print(f'got reward: {reward}')
            if reward == 30:
                print('completed task!')

            state = env.featurize_fn(env.base_env.state)[0]       
            print(sum(state))

            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break