import gym
from gym.spaces import Box
from gym.core import ObservationWrapper
from torchvision.transforms import Grayscale, Resize, Compose, Normalize
import numpy as np
import torch
class SkipFrame(gym.Wrapper):
    def __init__(self,env,skip):
        super().__init__(env)
        self._skip = skip
    def step(self,action): 
        total_reward = 0.0 
        for i in range(self._skip): 
            state, reward, done, trunk, info = self.env.step(action) 
            total_reward += reward 
            if done:
                break
        return state, total_reward, done, trunk, info

class GrayScaleObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2] # type: ignore
        self.observation_space = Box(low=0,high=255,shape=(1,*obs_shape),dtype=np.uint8)
    def permute_orientation(self,observation):
        observation = np.transpose(observation,(2,0,1))
        observation = torch.tensor(observation.copy(),dtype=torch.float)
        return observation 
    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transoform = Grayscale()
        observation = transoform(observation)
        return observation
class ResizeObservation(ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape,int):
            self.shape = (shape,shape)
        else:
            self.shape = tuple(shape)
        obs_shape = (self.observation_space.shape[0],) + self.shape # type: ignore
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    def observation(self, observation):
        transforms = Compose([Resize(self.shape,antialias=True),Normalize(mean=[0.5],std=[0.5])])
        observation = transforms(observation).squeeze(0)
        return observation
        
