# Irrigation Optimization with Deep Q-Learning

This project aims to optimize irrigation scheduling for a simulated farming field using Deep Q-Learning (DQN). The goal is to maintain optimal soil moisture and crop health by making intelligent decisions about watering specific areas, applying fertilizers, and adjusting irrigation schedules.

## Mission Statement
My mission is to cultivate a resilient farming community, empowering farmers to thrive year-round using technology, thereby fortifying food security for all.

## Project Overview
The project involves creating a custom Gym environment to simulate a farming field with various soil moisture levels and crops. A DQN agent is trained to optimize irrigation strategies based on the state of the field. Rewards are given for maintaining optimal conditions, and penalties are imposed for overwatering or underwatering.

## Files and Directories
- `irrigation_env.py`: Custom Gym environment for the irrigation optimization task.
- `train.py`: Script to train the DQN agent on the custom environment.
- `play.py`: Script to simulate the environment using the trained policy network.
- `requirements.txt`: List of required Python packages.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DQN.git
   cd DQN
2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install the required packages:
    ```bash
    pip install -r requirements.txt

## Custom Environment
The custom environment (IrrigationEnv) is defined in *irrigation_env.py*. It simulates a farming field with different soil moisture levels and crops.The agent can take actions such as watering specific areas, applying fertilizers, and adjusting irrigation schedules.

## Training the Agent
The training script (train.py) sets up the DQN agent using Keras-RL and trains it on the custom environment.

To train the agent, run: ```python train.py```

This will create and save the trained policy network in a file named *dqn_irrigation_weights.h5f*.

## Simulating Irrigation
The simulation script (play.py) uses the trained policy network to simulate the irrigation process. It demonstrates the agent's actions based on the optimized policy.

To run the simulation, execute: ```python play.py```

## Video Demonstration
A 5-minute video demonstration of the simulation can be found [here]().

## References
- [Gym Environment Creation](https://www.gymlibrary.dev/content/environment_creation/)
- [Keras-RL Documentation](https://keras-rl.readthedocs.io/en/latest/)