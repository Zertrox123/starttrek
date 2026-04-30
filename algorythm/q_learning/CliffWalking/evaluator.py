import gym
import numpy as np
import pickle as pkl

try:
    import cv2
except ImportError:
    cv2 = None

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

cliffEnv = gym.make("CliffWalking-v0", render_mode="rgb_array" if cv2 else "ansi")
q_table = pkl.load(open("q_learning.pkl", "rb"))

def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4))
    return action


NUM_EPISODES = 1
for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0
    state, _ = cliffEnv.reset()
    while not done:
        if cv2:
            cv2.imshow("Cliff Walking", cliffEnv.render())
            cv2.waitKey(250)
        else:
            print(cliffEnv.render())
        action = policy(state)
        state, reward, terminated, truncated, _ = cliffEnv.step(action)
        done = terminated or truncated
        total_reward += reward
        episode_length += 1
    print(f"Episode {episode} - Total reward: {total_reward} - Episode length: {episode_length}")
cliffEnv.close()
if cv2:
    cv2.destroyAllWindows()