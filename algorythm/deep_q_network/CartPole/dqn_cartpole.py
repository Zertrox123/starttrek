import gymnasium as gym
import tensorflow as tf
import pandas as pd
from keras import Model, Input
from keras.models import clone_model
from keras.layers import Dense
from keras.losses import Huber

env = gym.make("CartPole-v1")

net_input = Input(shape=(4,))
x = Dense(32, activation="relu")(net_input)
x = Dense(16, activation="relu")(x)
output = Dense(2, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)
q_net.compile(optimizer="adam")
loss_fn = Huber()

target_net = clone_model(q_net)

EPSILON = 1.0
EPSILON_DECAY = 1.005
GAMMA = 0.99
MAX_TRANSITION= 1_00_000
NUM_EPISODES = 300
REPLAY_BUFFER = []
BATCH_SIZE = 64
TARGET_UPDATE_AFTER = 1000
LEARN_AFTER_STEPS = 3

def insert_transition(transition):
    if len(REPLAY_BUFFER) >= MAX_TRANSITION:
        REPLAY_BUFFER.pop(0)
    REPLAY_BUFFER.append(transition)

def sample_transition(batch_size=16):
    random_indices = tf.random.uniform(shape=(batch_size, ), minval = 0, maxval = len(REPLAY_BUFFER), dtype = tf.int32)
    sampled_current_states = []
    sampled_actions = []
    sampled_rewards = []
    sampled_next_states = []
    sampled_terminals = []
    for index in random_indices:
        sampled_current_states.append(REPLAY_BUFFER[index][0])
        sampled_actions.append(REPLAY_BUFFER[index][1])
        sampled_rewards.append(REPLAY_BUFFER[index][2])
        sampled_next_states.append(REPLAY_BUFFER[index][3])
        sampled_terminals.append(REPLAY_BUFFER[index][4])
    return (
        tf.convert_to_tensor(sampled_current_states),
        tf.convert_to_tensor(sampled_actions),
        tf.convert_to_tensor(sampled_rewards),
        tf.convert_to_tensor(sampled_next_states),
        tf.convert_to_tensor(sampled_terminals),
    )

def policy(state, explore=0.0):
    action = tf.argmax(q_net(tf.expand_dims(state, axis = 0))[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval = 1) <= explore:
        action = tf.random.uniform(shape = (), maxval = 2, dtype = tf.int32)
    return action

def calculate_reward(state):
    reward = -1.0
    if -0.5 <= state[0] <= 0.5 and -1 <= state[1] <= 1 and -0.07 <= state[2] <= 0.07 and -0.525 <= state[2] <= 0.525:
        reward = 1.0
    return reward

random_states = []
done = False
state, _ = env.reset()
i = 0
while i < 20 and not done:
    random_states.append(state)
    state, _, terminated, truncated, _ = env.step(int(policy(state).numpy()))
    done = terminated or truncated
    i += 1

random_states = tf.convert_to_tensor(random_states)

def get_q_values(states):
    return tf.reduce_max(q_net(states), axis = 1)

step_counter = 0
metric = {"episode": [], "length": [], "total_reward": [], "avg_q": [], "exploration": []}
for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0
    state, _ = env.reset()
    while not done:
        action = policy(state, EPSILON)
        next_state, _, terminated, truncated, _ = env.step(int(action.numpy()))
        done = terminated or truncated
        reward = calculate_reward(next_state)
        insert_transition([state, action, reward, next_state, done])
        state = next_state
        step_counter += 1
        if len(REPLAY_BUFFER) < BATCH_SIZE:
            total_reward += reward
            episode_length += 1
            continue

        if step_counter % LEARN_AFTER_STEPS == 0:
            current_states, actions, rewards, next_states, terminals = sample_transition(BATCH_SIZE)
            next_action_values = tf.reduce_max(target_net(next_states), axis = 1)
            targets = tf.where(terminals, rewards, rewards + GAMMA * next_action_values)

            with tf.GradientTape() as tape:
                preds = q_net(current_states)
                batch_nums = tf.range(0, limit=BATCH_SIZE)
                indices = tf.stack((batch_nums, actions), axis = 1)
                current_values = tf.gather_nd(preds, indices)
                loss = loss_fn(targets, current_values)
            grads = tape.gradient(loss, q_net.trainable_weights)
            q_net.optimizer.apply_gradients(zip(grads, q_net.trainable_weights))

        if step_counter % TARGET_UPDATE_AFTER == 0:
            target_net.set_weights(q_net.get_weights())
        total_reward += reward
        episode_length += 1

    avg_q = tf.reduce_mean(get_q_values(random_states)).numpy()
    metric["episode"].append(episode)
    metric["length"].append(episode_length)
    metric["total_reward"].append(total_reward)
    metric["avg_q"].append(tf.reduce_mean(get_q_values(random_states)).numpy())
    metric["exploration"].append(EPSILON)
    EPSILON /= EPSILON_DECAY
    pd.DataFrame(metric).to_csv("metric.csv", index=False)
env.close()
q_net.save("dqn_q_net.keras")