import gymnasium as gym  # type: ignore
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym.make("SuperMarioBros-v0", render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
print("Hello")
for step in range(5000):
    if done:
        state, info = env.reset(seed=123)

    state, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    
    done = terminated or truncated
    env.render()

env.close()