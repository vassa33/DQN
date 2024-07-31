import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
from irrigation_env import IrrigationEnv

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10,
                   target_model_update=1e-2)
    return dqn

def evaluate_agent(dqn, env, episodes=10):
    dqn.test(env, nb_episodes=episodes, visualize=False)

def main():
    env = IrrigationEnv()
    states = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(states, actions)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    # Load existing weights if available
    if os.path.exists('dqn_irrigation_weights.h5f'):
        dqn.load_weights('dqn_irrigation_weights.h5f')

    # Set up checkpointing
    checkpoint_callback = ModelIntervalCheckpoint('dqn_checkpoint_{step}.h5f', interval=10000)

    # Train the agent with checkpointing
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1, callbacks=[checkpoint_callback])

    # Save the final weights
    dqn.save_weights('dqn_irrigation_weights.h5f', overwrite=True)

    # Evaluate the agent
    evaluate_agent(dqn, env)

if __name__ == "__main__":
    main()
