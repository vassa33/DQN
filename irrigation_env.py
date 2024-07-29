import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class IrrigationEnv(gym.Env):
    def __init__(self, field_size=5):
        super(IrrigationEnv, self).__init__()
        
        self.field_size = field_size
        self.moisture_levels = np.zeros((field_size, field_size))
        self.crop_health = np.zeros((field_size, field_size))
        
        # Define action space: water, fertilize, adjust schedule for each cell
        self.action_space = spaces.Discrete(3 * field_size * field_size)
        
        # Define observation space: moisture levels and crop health for each cell
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=(2, field_size, field_size), 
                                            dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = 100  # One growing season
        
    def reset(self):
        self.moisture_levels = np.random.uniform(0.3, 0.7, (self.field_size, self.field_size))
        self.crop_health = np.random.uniform(0.5, 1.0, (self.field_size, self.field_size))
        self.current_step = 0
        return self._get_observation()
    
    def step(self, action):
        action_type = action // (self.field_size * self.field_size)
        cell = action % (self.field_size * self.field_size)
        row, col = divmod(cell, self.field_size)
        
        if action_type == 0:  # Water
            self.moisture_levels[row, col] = min(1.0, self.moisture_levels[row, col] + 0.1)
        elif action_type == 1:  # Fertilize
            self.crop_health[row, col] = min(1.0, self.crop_health[row, col] + 0.05)
        elif action_type == 2:  # Adjust schedule (simulated by small moisture change)
            self.moisture_levels[row, col] = max(0, min(1.0, self.moisture_levels[row, col] + np.random.uniform(-0.05, 0.05)))
        
        # Natural changes
        self.moisture_levels -= 0.02  # Evaporation
        self.moisture_levels = np.clip(self.moisture_levels, 0, 1)
        self.crop_health += np.where((self.moisture_levels > 0.4) & (self.moisture_levels < 0.8), 0.01, -0.02)
        self.crop_health = np.clip(self.crop_health, 0, 1)
        
        # Calculate reward
        reward = np.mean(self.crop_health) - 0.1 * np.abs(0.6 - np.mean(self.moisture_levels))
        
        self.current_step += 1
        done = self.current_step >= self.max_steps or np.all(self.crop_health >= 0.9)
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        return np.stack([self.moisture_levels, self.crop_health])
    
    def render(self, mode='human'):
        if mode == 'rgb_array':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            moisture_cmap = LinearSegmentedColormap.from_list("", ["red", "yellow", "green", "blue"])
            health_cmap = LinearSegmentedColormap.from_list("", ["brown", "yellow", "green"])
            
            im1 = ax1.imshow(self.moisture_levels, cmap=moisture_cmap, vmin=0, vmax=1)
            ax1.set_title("Moisture Levels")
            fig.colorbar(im1, ax=ax1)
            
            im2 = ax2.imshow(self.crop_health, cmap=health_cmap, vmin=0, vmax=1)
            ax2.set_title("Crop Health")
            fig.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return image
        elif mode == 'human':
            print(f"Step: {self.current_step}")
            print("Moisture Levels:")
            print(self.moisture_levels)
            print("Crop Health:")
            print(self.crop_health)