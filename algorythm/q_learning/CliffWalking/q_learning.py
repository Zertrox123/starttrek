import gym
import numpy as np
import pickle as pkl

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

cliffEnv = gym.make("CliffWalking-v0")
q_table = np.zeros(shape=(48, 4))

# Parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 1000

def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4))
    return action

for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0
    state, _ = cliffEnv.reset()
    while not done:
        action = policy(state, EPSILON)
        next_state, reward, terminated, truncated, _ = cliffEnv.step(action)
        done = terminated or truncated
        next_action = policy(next_state, EPSILON)   
        q_table[state][action] += ALPHA * (reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])
        reward -= 1
        state = next_state
        action = next_action
        total_reward += reward
        episode_length += 1
    print(f"Episode {episode} - Total reward: {total_reward} - Episode length: {episode_length}")
cliffEnv.close()
pkl.dump(q_table, open("q_learning.pkl", "wb"))