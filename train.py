import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from irrigation_env import IrrigationEnv

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10,
                   target_model_update=1e-2)
    return dqn

def main():
    env = IrrigationEnv()
    states = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(states, actions)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Train the agent
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    # Save the final weights
    dqn.save_weights('dqn_irrigation_weights.h5f', overwrite=True)

if __name__ == "__main__":
    main()