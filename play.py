import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
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

def run_episode(env, agent):
    frames = []
    observation = env.reset()
    done = False
    while not done:
        frames.append(env.render(mode='rgb_array'))
        action = agent.forward(observation)
        observation, reward, done, _ = env.step(action)
    return frames

def main():
    env = IrrigationEnv()
    states = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(states, actions)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Load the trained weights
    dqn.load_weights('dqn_irrigation_weights.h5f')

    # Run an episode and collect frames
    frames = run_episode(env, dqn)

    # Create animation
    fig = plt.figure(figsize=(12, 5))
    ani = FuncAnimation(fig, lambda i: plt.imshow(frames[i]), frames=len(frames), interval=200)

    # Save animation as mp4
    Writer = writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('irrigation_simulation.mp4', writer=writer)

    print("Simulation video saved as 'irrigation_simulation.mp4'")

if __name__ == "__main__":
    main()