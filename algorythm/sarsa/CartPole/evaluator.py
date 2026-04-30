import gymnasium as gym, cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

env = gym.make("CartPole-v1", render_mode="rgb_array")
q_net = load_model("sarsa_q_net.keras")

def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    return action

for episode in range(5):
    done = False
    state, _ = env.reset()
    state = tf.convert_to_tensor([state], dtype=tf.float32)
    while not done:
        frame = env.render()
        cv2.imshow("cartPole", frame)
        cv2.waitKey(100)
        action = policy(state)
        state, reward, terminated, truncated, _ = env.step(int(action.numpy()))
        done = terminated or truncated
        state = tf.convert_to_tensor([state], dtype=tf.float32)
env.close()